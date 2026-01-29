import torch
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.base import BaseModel
from src.models.sasrec import (
    SASRecModel,
    collate_fn_sasrec,
)
from src.models.ultragcn import UltraGCN


class JointSASRecUltraGCN(BaseModel, nn.Module):

    def __init__(
        self,
        name: str = "mix_model",
        seed: int = 42,
        device: str = "cuda",

        
        hidden_size: int = 128,
        lr: float = 1e-3,
        lambda_reg: float = 1e-4,
        n_epochs: int = 1000,

        
        lambda_sas: float = 1.0,
        lambda_ultra: float = 0.1,

        
        seq_len: int = 200,
        num_blocks: int = 3,
        num_heads: int = 2,
        dropout: float = 0.2,
        batch_size_sas: int = 256,
        log_step: int = 100,
        mix_x: bool = True,
        alpha: float = 1.0,
        beta: float = 16.0,
        bucket_size_y: int = 256,
        s: int = 1024,

        
        embedding_dim: int = 128,
        w1: float = 1e-7,
        w2: float = 1.0,
        w3: float = 1e-7,
        w4: float = 1.0,
        negative_weight: float = 200,
        negative_num: int = 200,
        gamma: float = 1e-3,
        num_neighbors: int = 10,
    ):
        
        BaseModel.__init__(self, name)
        nn.Module.__init__(self)

        
        self.device = torch.device(device)
        torch.manual_seed(seed)

        self.hidden_size = hidden_size
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.n_epochs = n_epochs

        self.lambda_sas = lambda_sas
        self.lambda_ultra = lambda_ultra

        
        self.seq_len = seq_len
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_size = batch_size_sas
        self.log_step = log_step
        self.mix_x = mix_x
        self.alpha = alpha
        self.beta = beta
        self.bucket_size_y = bucket_size_y
        self.s = s

        
        self.embedding_dim = embedding_dim
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.negative_weight = negative_weight
        self.negative_num = negative_num
        self.gamma = gamma
        self.num_neighbors = num_neighbors

        
        
        self.sasrec = SASRecModel(
            name="sasrec",
            hidden_size=self.hidden_size,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=self.dropout,
            lr=self.lr,
            device=self.device.type,
            seq_len=self.seq_len,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            mix_x=self.mix_x,
            alpha=self.alpha,
            beta=self.beta,
            bucket_size_y=self.bucket_size_y,
            s=self.s,
            log_step=self.log_step,
        )

        
        self.ultragcn = UltraGCN(
            name="ultragcn",
            embedding_dim=self.embedding_dim,
            w1=self.w1,
            w2=self.w2,
            w3=self.w3,
            w4=self.w4,
            negative_weight=self.negative_weight,
            negative_num=self.negative_num,
            gamma=self.gamma,
            lambda_reg=self.lambda_reg,
            num_epochs=self.n_epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            num_neighbors=self.num_neighbors,
            early_stopping=False,  
        )

        
    
    

    def _post_init(self, train_dataset):
        
        self.n_users = train_dataset.n_users
        self.n_items = train_dataset.n_items

        
        
        if hasattr(self.sasrec, "_post_init"):
            self.sasrec._post_init(train_dataset, None)
        else:
            
            raise RuntimeError("SASRec model does not implement _post_init required by joint trainer")

        
        self.ultragcn.n_users = self.n_users
        self.ultragcn.n_items = self.n_items
        self.ultragcn.device = self.device

        
        self.ultragcn.user_embeddings = nn.Embedding(self.n_users, self.sasrec.hidden_size).to(self.device)
        nn.init.xavier_uniform_(self.ultragcn.user_embeddings.weight)

        
        
        self.item_embeddings = self.sasrec.item_emb
        self.ultragcn.item_embeddings = self.item_embeddings

        
        coo = train_dataset.get_coo_array()
        self.ultragcn.user_betas, self.ultragcn.item_betas = self.ultragcn._compute_betas(coo)
        self.ultragcn.constraint_matrix, self.ultragcn.neighbor_matrix = self.ultragcn._compute_item_item_matrix(coo, dataset_name=train_dataset.name)

        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.lambda_reg)

        
        self.to(self.device)

    
    
    

    def fit(self, train_dataset, val_dataset=None):
        
        self._post_init(train_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=lambda x: collate_fn_sasrec(x, self.sasrec.seq_len, self.sasrec.item_num, True)
        )

        step = 0

        for epoch in range(self.n_epochs):
            self.train()
            epoch_loss = []

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.n_epochs}"):

                self.optimizer.zero_grad()

                users = batch["user_id"].to(self.device)
                seq = batch["seq"].to(self.device)
                pos = batch["pos"].to(self.device)

                
                sas_out = self.sasrec._sasrec_forward(seq, pos)
                if isinstance(sas_out, tuple) and len(sas_out) == 2:
                    loss_sas, _ = sas_out
                else:
                    loss_sas = sas_out

                
                pos_items = pos[:, -1].to(self.device)
                
                pad_token = self.sasrec.item_num
                valid_mask = pos_items != pad_token

                if valid_mask.sum() == 0:
                    loss_ultra = torch.tensor(0.0, device=self.device)
                else:
                    ultra_users = users[valid_mask]
                    ultra_pos_items = pos_items[valid_mask]
                    neg_items = torch.randint(0, self.ultragcn.n_items, (len(ultra_users), self.ultragcn.negative_num), device=self.device)
                    omega = self.ultragcn.get_omegas(ultra_users, ultra_pos_items, neg_items)
                    loss_L = self.ultragcn.cal_loss_L(ultra_users, ultra_pos_items, neg_items, omega)
                    loss_I = self.ultragcn.cal_loss_I(ultra_users, ultra_pos_items)
                    loss_ultra = loss_L + self.ultragcn.gamma * loss_I

                
                loss = self.lambda_sas * loss_sas + self.lambda_ultra * loss_ultra

                loss.backward()
                self.optimizer.step()

                epoch_loss.append(loss.item())
                step += 1

                if step % self.log_step == 0:
                    print(f"[step {step}] recent mean loss = {np.mean(epoch_loss[-self.log_step:]):.6f}")

            print(f"Epoch {epoch+1} mean loss: {np.mean(epoch_loss):.6f}")

    
    
    

    @torch.no_grad()
    def predict(self, dataset, top_n: int):
        return self.sasrec.predict(dataset, top_n)

    def suggest_additional_params(self):
        return {"num_epochs": int(self.n_epochs)}

