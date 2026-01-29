from src.base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import random
import time
import uuid
from collections import defaultdict
from datetime import datetime
import pandas as pd
from tqdm import trange, tqdm
from typing import Union

import time

from torch.utils.data import DataLoader
from src.metrics import NDCGMetric, RecallMetric

def fix_torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pad_seqs(user_items, maxlen, pad_token):
    seq = np.full(maxlen, pad_token, dtype=np.int32)
    pos = np.full(maxlen, pad_token, dtype=np.int32)
    neg = np.empty((maxlen, 1))

    n_user_items = min(len(user_items) - 1, maxlen)

    if n_user_items > 0:
        seq[-n_user_items:] = user_items[-n_user_items-1:-1]
        pos[-n_user_items:] = user_items[-n_user_items:]

    return seq, pos, neg
    
def collate_fn_sasrec(x, maxlen, pad_token, is_train):

    seqs_batch = []
    pos_batch = []
    user_ids = []
    seen_history = []

    for user in x:
        seq, pos, neg = pad_seqs(user["history"].tolist(), maxlen, pad_token)

        seqs_batch.append(seq)
        pos_batch.append(pos)
        user_ids.append(user["user_id"])
        if not is_train:
            seen_history.append(user["history"])

    batch = {
        "seq": torch.LongTensor(np.asarray(seqs_batch)),
        "pos": torch.LongTensor(np.asarray(pos_batch)),
        "user_id": torch.LongTensor(np.asarray(user_ids)),
        "seen_history": torch.LongTensor(np.asarray(seen_history))
    }

    return batch


def pad_history(itemlist, length, pad_item):
    if len(itemlist) >= length:
        return itemlist[-length:]
    if len(itemlist) < length:
        temp = [pad_item] * (length - len(itemlist))
        temp.extend(itemlist)
        return temp


class PointWiseFeedForward(torch.nn.Module):

    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.ff = nn.Sequential(
            torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1),
            torch.nn.Dropout(p=dropout_rate)
        )

    def forward(self, inputs):
        outputs = self.ff(
            inputs.transpose(-1,-2)
        )
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs

class SASRecBackBone(nn.Module):
    def __init__(
            self,
            item_num,
            hidden_units,
            dropout_rate,
            maxlen,
            num_blocks,
            num_heads,
            manual_seed=37
        ):
        super(SASRecBackBone, self).__init__()
        self.item_num = item_num
        self.pad_token = item_num

        self.item_emb = nn.Embedding(self.item_num+1, hidden_units, padding_idx=self.pad_token)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(p=dropout_rate)

        self.attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer =  nn.MultiheadAttention(
                hidden_units,num_heads,dropout_rate
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        fix_torch_seed(manual_seed)
        self.initialize()

    def initialize(self):
        for _, param in self.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except:
                pass # just ignore those failed init layers

    def log2feats(self, log_seqs):
        device = log_seqs.device
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.arange(log_seqs.shape[1]), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs == self.pad_token
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.full((tl, tl), True, device=device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)
        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats[:, None, :, :] * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def score(self, seq):
        '''
        Takes 1d sequence as input and returns prediction scores.
        '''
        log_feats = self.log2feats(seq.unsqueeze(0))
        final_feat = log_feats[:, -1, :] # only use last QKV classifier

        item_embs = self.item_emb.weight
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits


class SASRecModel(BaseModel, SASRecBackBone):
    def __init__(
        self,
        name: str = "sasrec",
        hidden_size: int = 64,
        num_blocks: int = 1,
        num_heads: int = 1,
        dropout: float = 0.1,
        lr: float = 5e-3,
        device: str = "cuda",
        seq_len: int = 50,
        n_epochs: int = 2000,
        batch_size: int = 64,
        seed: int = 52,
        log_step: int = 200,
        mix_x: bool = True,
        alpha: float = 1.0,
        beta: float = 16.0,
        bucket_size_y: int = 256,
        s: int = 1024
    ):
        BaseModel.__init__(self, name)

        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lr = lr
        self.device = device
        self.seq_len = seq_len
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_heads = num_heads

        self.alpha = alpha
        self.beta = beta
        self.s = s

        self.mix_x = mix_x

        self.bucket_size_y = bucket_size_y

        self.log_step = log_step

        self.val_top_ns = [5, 10, 20]
    
    def _post_init(self, train_dataset, val_dataset):
        self.n_items = train_dataset.n_items

        len_list = []

        for user in train_dataset:
            len_list.append(len(user["history"]))

        mean_len = sum(len_list) / len(len_list)

        self.bucket_size_x = int(self.alpha * np.sqrt(self.s * mean_len * self.beta))
        self.n_buckets = int(self.alpha * np.sqrt(self.s * self.seq_len / self.beta))

        print("Calculated bucket size:", self.bucket_size_x)
        print("Calculated n buckets:", self.n_buckets)

        SASRecBackBone.__init__(
            self,
            self.n_items + 1,
            self.hidden_size,
            self.dropout,
            self.seq_len,
            self.num_blocks,
            self.num_heads,
            manual_seed=37
        )

        self.to(self.device)

        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))

    def _sasrec_forward(self, log_seqs, pos_seqs):
        """
        Возвращает (loss, final_user_embedding).
        final_user_embedding shape: (batch_size, hidden_dim)
        """
        emb = self.log2feats(log_seqs)   # (B, T, D)
        hd = emb.shape[-1]

        # --- existing sampled-CE logic (не трогаем) ---
        x = emb.view(-1, hd)
        y = pos_seqs.view(-1)
        w = self.item_emb.weight

        correct_class_logits_ = (x * torch.index_select(w, dim=0, index=y)).sum(dim=1) # (bs,)

        with torch.no_grad():
            if self.mix_x:
                omega = 1/np.sqrt(np.sqrt(hd)) * torch.randn(x.shape[0], self.n_buckets, device=x.device)
                buckets = omega.T @ x
                del omega
            else:
                buckets = 1/np.sqrt(np.sqrt(hd)) * torch.randn(self.n_buckets, hd, device=x.device) # (n_b, hd)

        with torch.no_grad():
            x_bucket = buckets @ x.T
            x_bucket[:, log_seqs.view(-1) == self.pad_token] = float('-inf')
            _, top_x_bucket = torch.topk(x_bucket, dim=1, k=self.bucket_size_x)
            del x_bucket

            y_bucket = buckets @ w.T
            y_bucket[:, self.pad_token] = float('-inf')
            _, top_y_bucket = torch.topk(y_bucket, dim=1, k=self.bucket_size_y)
            del y_bucket

        x_bucket = torch.gather(x, 0, top_x_bucket.view(-1, 1).expand(-1, hd)).view(self.n_buckets, self.bucket_size_x, hd)
        y_bucket = torch.gather(w, 0, top_y_bucket.view(-1, 1).expand(-1, hd)).view(self.n_buckets, self.bucket_size_y, hd)
        
        wrong_class_logits = (x_bucket @ y_bucket.transpose(-1, -2))
        mask = torch.index_select(y, dim=0, index=top_x_bucket.view(-1)).view(self.n_buckets, self.bucket_size_x)[:, :, None] == top_y_bucket[:, None, :]
        wrong_class_logits = wrong_class_logits.masked_fill(mask, float('-inf'))
        correct_class_logits = torch.index_select(correct_class_logits_, dim=0, index=top_x_bucket.view(-1)).view(self.n_buckets, self.bucket_size_x)[:, :, None]
        logits = torch.cat((wrong_class_logits, correct_class_logits), dim=2)

        loss_ = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            (logits.shape[-1] - 1) * torch.ones(logits.shape[0] * logits.shape[1], dtype=torch.int64, device=logits.device),
            reduction='none'
        )
        loss = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        loss.scatter_reduce_(0, top_x_bucket.view(-1), loss_, reduce='amax', include_self=False)
        loss = loss[(loss != 0) & (y != self.pad_token)]
        loss = torch.mean(loss)

        # --- final user embedding (последний временной шаг) ---
        final_user_emb = emb[:, -1, :]   # shape (B, D)

        return loss, final_user_emb
            
    
    
    def fit(self, train_dataset, val_dataset):
        """
        Fit the model to the dataset.
        """

        self._post_init(
            train_dataset,
            val_dataset
        )

        train_dataloader = DataLoader(
            train_dataset,
            self.batch_size,
            shuffle = True,
            num_workers = 10,
            collate_fn = lambda x: collate_fn_sasrec(x, self.seq_len, self.item_num, True)
        )

        for i in trange(self.n_epochs):
            self.train()

            train_loss = []

            index = 0

            for batch in tqdm(train_dataloader):

                self.opt.zero_grad()

                loss, _ = self._sasrec_forward(
                    batch["seq"].to(self.device),
                    batch["pos"].to(self.device),
                )

                train_loss.append(loss.item())

                loss.backward()
                
                self.opt.step() 

                index += 1

                if (index + 1) % self.log_step == 0:
                    print(f"Mean train loss: {sum(train_loss[-self.log_step:]) / self.log_step}")    
            
            
            print(f"Mean train loss: {sum(train_loss) / len(train_loss)}")

            if val_dataset is not None and (i + 1) % 50 == 0:
                holdout_users = val_dataset.get_holdout_users()

                for val_top_n in self.val_top_ns:

                    predictions = self.predict(
                        val_dataset,
                        val_top_n
                    )

                    for metric_class in [NDCGMetric, RecallMetric]:
                        
                        metric = metric_class(val_top_n)
                        val_metric = metric(
                            predictions[holdout_users, :],
                            val_dataset.get_holdout_array()[holdout_users],
                        )
                        val_metric_f = float(val_metric)

                        print(f"Metric {metric.name}: {val_metric_f}")
        
    @torch.no_grad()
    def predict(self, dataset, top_n: int, batch_size: int = 1024):
        self.eval()

        n_users = dataset.n_users
        
        dataloader = DataLoader(
            dataset,
            1,
            shuffle = False,
            num_workers = 10,
            collate_fn = lambda x: collate_fn_sasrec(x, self.seq_len, self.item_num, False)
        )
    
        recommendations = np.zeros((n_users, top_n))

        for batch in dataloader:
    
            logits = self.score(
                seq=batch["seq"][0, :].to(self.device),
            )

            logits[:, batch["seen_history"][0]] = -10 ** 9
    
            top_items = torch.topk(
                logits,
                k=top_n,
                dim=1
            ).indices
    
            for uid, recs in zip(batch["user_id"], top_items):
                recommendations[uid, :] = recs.cpu().numpy()
                
        return recommendations
    
    