# Self-Supervised Graph Transformer with Atom-to-Bond Attention for Molecular Property Prediction

This repository is the official implementation of **MolPGT**.

## Environments

### Install via Conda

```bash
# Clone the environment
conda env create -f env.yml
# Activate the environment
conda activate molpgt
```
## Data Preparation

We provide pre-processed datasets download address: https://pan.quark.cn/s/6295350998d4

You can also preprocess the dataset yourself by running the following command.

```bash
# pretrain
python molpgt/data/data_preprocess.py --base_path datasets --output pretrain --val_num 25000

# finetune
python molpgt/data/finetune_preprocess.py --base_path datasets --output finetune --dataset bace
```

The raw datasets are listed as follows:

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

please run the following command:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python pretrain_cl.py --config_path config/pretrain_cl.yml
```

## Fine-tuning

If you want to use our pre-trained model directly for molecular property prediction, please run the following command:

```bash
# For classification tasks
python finetune_classificaiton.py --config_path config/finetune_classificaiton.yml
# For regression tasks
python finetune_reg.py --config_path config/finetune_reg.yml
```
The following table gives the hyperparameter selection for each task:

| Dataset       | BACE | ClinTox | Tox21 | ToxCast | SIDER | HIV  | MUV  | ESOL | FreeSolv | Lipo |
|---------------|------|------|-------|-------|-------|------|------|------|-------|------|       
| task_num      | 1    | 2    | 12    | 617   | 27    | 1    | 17   | 1    | 1     | 1    |
| lr            | 4e-4 | 1e-4 | 4e-4  | 1e-4  | 1e-4  | 1e-4 | 1e-4 | 1e-4 | 1e-4  | 1e-4 |
| max_lr        | 1e-3 | 1e-3 | 8e-4  | 5e-4  | 1e-3  | 1e-3 | 1e-3 | 1e-3 | 1e-3  | 5e-4 |
| final_lr      | 1e-4 | 1e-4 | 1e-4  | 1e-4  | 1e-4  | 1e-4 | 1e-4 | 1e-4 | 1e-4  | 1e-4 |
| batch_size    | 128  | 16   | 16    | 16    | 4     | 16   | 32   | 8    | 128   | 16   |
| epoch         | 5    | 10   | 15    | 25    | 30    | 5    | 15   | 65   | 35    | 35   |
| warmup_epochs | 2    | 2    | 5     | 5     | 2     | 3    | 2    | 3    | 2     | 2    |
| dropout       | 0    | 0    | 0     | 0.2   | 0.1   | 0    | 0    | 0    | 0.2   | 0    |
| n_gpus        | 1    | 1    | 2     | 4     | 2     | 4    | 6    | 2    | 1     | 4    |