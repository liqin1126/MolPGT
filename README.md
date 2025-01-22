# Self-Supervised Graph Transformer with Global Node-Edge Communications for Molecular Property Prediction

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

please run the following command to preprocess the pre-training dataset:

```bash
python molpgt/data/data_preprocess.py
```

please run the following command to preprocess the fine-tuning dataset:

```bash
python molpgt/data/finetune_preprocess.py
```

The raw data sets are listed as follows:

```
|-- datasets
    |-- raw
        |-- bace.csv
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
