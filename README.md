# Self-supervised graph transformer with global node-edge communications for molecular property prediction

This repository is the official implementation of **MolPGT**.

## Environments

Key dependencies and versions:

| Package | Version |
|---|---|
| Python | 3.8.17 |
| PyTorch | 1.12.1 (CUDA 11.3) |
| torchvision | 0.13.1 |
| torch-geometric | 2.3.1 |
| torch-scatter | 2.1.0+pt112cu113 |
| torch-sparse | 0.6.15+pt112cu113 |
| torch-cluster | 1.6.0+pt112cu113 |
| rdkit | 2022.9.5 |
| numpy | 1.21.6 |
| scikit-learn | 1.3.2 |
| pandas | 2.0.3 |
| scipy | 1.10.1 |
| huggingface-hub | 0.16.4 |
| lmdb | 1.4.1 |
| ml-collections | 0.1.1 |
| unicore | 0.0.1 |

### Install via Conda

```bash
# Create the conda environment, skip pip section
conda env create -f env.yml --no-pip

# Manually install PyG packages from the official wheel index
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-1.12.1+cu113.html

# Install remaining pip packages manually
pip install rdkit==2022.9.5 numpy==1.21.6 scikit-learn==1.3.2 \
    pandas==2.0.3 scipy==1.10.1 huggingface-hub==0.16.4 \
    lmdb==1.4.1 ml-collections==0.1.1
```

## Data Preparation

### Download Pre-processed Datasets

We provide pre-processed datasets on Hugging Face Hub for direct use:

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="tanlq/MolPGT-datasets",
    repo_type="dataset",
    local_dir="./datasets"
)
```

### Preprocess Datasets from Scratch

You can also preprocess the raw datasets yourself by running the following commands:

```bash
# Pre-training
python molpgt/data/data_preprocess.py --base_path datasets --output pretrain --val_num 25000

# Fine-tuning (e.g., bace)
python molpgt/data/finetune_preprocess.py --base_path datasets --output finetune --dataset bace
```

The raw datasets should be organized as follows:

```
|-- datasets
    |-- raw
        |-- bace.csv
        |-- ...
```

The processed datasets are listed as follows:

```
|-- datasets
    |-- pretrain
        |-- train_block.pkl
        |-- val_block.pkl
        |-- summary.json
    |-- finetune
        |-- bace.pkl
        |-- bace_summary.json
        |-- esol.pkl
        |-- ...
```

## Model

The generated pre-training model are listed as follows:

```
|-- checkpoints
    |-- pretrain
        |-- checkpoint98
```

## Pre-training

Please run the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 script/pretrain_cl.py --config_path config/pretrain_cl.yml
```

## Fine-tuning

Please run the following commands based on the required number of GPUs for each task (refer to the `n_gpus` row in the hyperparameter table below):

```bash
# For classification tasks
CUDA_VISIBLE_DEVICES=<gpu_ids> python -m torch.distributed.launch --nproc_per_node=<n_gpus> script/finetune_classification.py --config_path config/finetune_classification.yml

# For regression tasks
CUDA_VISIBLE_DEVICES=<gpu_ids> python -m torch.distributed.launch --nproc_per_node=<n_gpus> script/finetune_reg.py --config_path config/finetune_reg.yml
```

For example, to fine-tune on **HIV** (requires 4 GPUs):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 script/finetune_classification.py --config_path config/finetune_classification.yml
```

### Hyperparameter Selection

| Dataset       | BACE | ClinTox | Tox21 | ToxCast | SIDER | HIV  | MUV  | ESOL | FreeSolv | Lipo |
|---------------|------|---------|-------|---------|-------|------|------|------|----------|------|
| task_num      | 1    | 2       | 12    | 617     | 27    | 1    | 17   | 1    | 1        | 1    |
| lr            | 4e-4 | 1e-4    | 4e-4  | 1e-4    | 1e-4  | 1e-4 | 1e-4 | 1e-4 | 1e-4     | 1e-4 |
| max_lr        | 1e-3 | 1e-3    | 8e-4  | 5e-4    | 1e-3  | 1e-3 | 1e-3 | 1e-3 | 1e-3     | 5e-4 |
| final_lr      | 1e-4 | 1e-4    | 1e-4  | 1e-4    | 1e-4  | 1e-4 | 1e-4 | 1e-4 | 1e-4     | 1e-4 |
| batch_size    | 128  | 16      | 16    | 16      | 4     | 16   | 32   | 8    | 128      | 16   |
| epoch         | 5    | 10      | 15    | 25      | 30    | 5    | 15   | 65   | 35       | 35   |
| warmup_epochs | 2    | 2       | 5     | 5       | 2     | 3    | 2    | 3    | 2        | 2    |
| dropout       | 0    | 0       | 0     | 0.2     | 0.1   | 0    | 0    | 0    | 0.2      | 0    |
| n_gpus        | 1    | 1       | 2     | 4       | 2     | 4    | 6    | 2    | 1        | 4    |
