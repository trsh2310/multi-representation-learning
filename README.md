# Leveraging multiple representations of behavioral data for training recommender systems

## Datasets

To make global time split and random holdout use this:

```bash
python3 scripts/dataset_pipeline.py --filename <path_to_csv>
--user_col <name_of_userid_column>
--item_col <name_of_itemid_column>
--time_col <name_of_timestamp_column>
--rating_col <name_of_rating_column>
```

### parse_datasets_yd

Use this script to download raw CSVs from Yandex Disk, build splits, and generate dataset configs:

```bash
python3 parse_datasets_yd.py --splits_public_url <public_folder_with_splits>
```

After the pipeline finishes, upload the resulting split folders to disk and make the folder public. Then pass that public folder URL via `--splits_public_url` so the generated configs point to the correct location.

## Training

To train a model with a specific configuration, use the following command.  
The configuration file should be placed inside the `configs/` folder.

```bash
python3 run_model.py -cn <your_config>
```

You can also override individual variables directly from the command line:

```bash
python3 run_model.py -cn <your_config> var1=32 var2=32
```

## Hyperparameter Optimization

To run hyperparameter optimization, use the `run_optuna.py` script:

Before running experiments, make sure datasets are downloaded:

```bash
python3 offline_download.py
```

```bash
python3 run_optuna.py --config_name <config_name> \
	--dataset <dataset_name> \
	--optuna_params <optuna_config> \
	--experiment_name <experiment_name>
```

Arguments:

* `--config_name` (`-cn`): name of the base configuration file used for the experiments
* `--dataset` (`-ds`): name of dataset to train on
* `--optuna_params` (`-op`): name of optuna config (see configs/optuna_params)
* `--experiment_name` (`-en`): name of your experiment (should be unique)

## Batch runs (sbatch) and run script

The wrapper script [sbatch/run.py](sbatch/run.py) runs `run_optuna.py` and selects a dataset by index from `data/metadata.json`.

**Submit sbatch jobs from the project root (the folder that contains `run_optuna.py`).**

Make sure to run `offline_download.py` before launching experiments (examples below already include it).

The script calls `offline_download.py`, reads `data/<dataset>/info.json`, sorts by $median_{interactions} \times num_{users}$, and writes the list to `data/metadata.json`.

### Local run

```bash
python3 sbatch/run.py --dataset_idx 1 --model <model_name> --optuna_params <optuna_params> --num_trials 200
```

### Using sbatch arrays

In a SLURM array, the index is taken from `SLURM_ARRAY_TASK_ID` and is expected to be 1-based.

Example sbatch snippet:

```bash
#SBATCH --array=1-10

python3 sbatch/run.py \
	--model <model_name> \
	--optuna_params <optuna_params> \
	--num_trials 200
```

If `data/metadata.json` is missing, `run.py` will automatically call `scripts/build_metadata.py`.
