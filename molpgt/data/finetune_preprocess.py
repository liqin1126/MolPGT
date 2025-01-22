import random

import os
import ast
import math
import numpy
import pickle
import lmdb
import copy
import json
import csv
from collections import defaultdict

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

import sys
from torch_geometric.data import Data, Dataset

from rdkit import RDLogger
from graph import mol_to_geograph_data_MMFF3d, mol_to_geograph_data
RDLogger.DisableLog('rdApp.*')
import argparse
from multiprocessing import Pool, set_start_method
set_start_method("spawn", force=True)

lmdb_path = './data/finetune'

def gen_geograph_data_from_smiles(raw_smiles, raw_qm_data, dataset):
    if dataset in ['qm7', 'qm8', 'qm9']:
        smiles = raw_qm_data['smi']
        mol = AllChem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        if mol is None:
            return None
        atom_poses = torch.tensor(numpy.array(raw_qm_data['coordinates']), dtype=torch.float32).view(-1, 3)
        data = mol_to_geograph_data(mol, atom_poses=atom_poses, smiles=smiles)
    else:
        smiles = raw_smiles
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            return None
        data = mol_to_geograph_data_MMFF3d(mol, smiles=smiles)
    if data is not None:
        data['smiles'] = smiles
    return data

def load_data_from_smiles(smiles_list, target_list, dataset):
    qm_data = []
    if dataset in ['bbbp']:
        re = []
        res = []
        for i, f in enumerate(smiles_list):
            if i % 1000 == 0:
                print('processed %d smiles' %i)
            smiles = f
            qm_data = None
            if qm_data is not None:
                raw_qm_data = qm_data[i]
            else:
                raw_qm_data = None
            data = gen_geograph_data_from_smiles(raw_smiles=smiles, raw_qm_data=raw_qm_data, dataset=dataset)
            if data is None:
                continue
            if dataset in ['qm7', 'qm8', 'qm9']:
                data.y = torch.tensor(raw_qm_data['target'], dtype=torch.float32)
            else:
                data.y = target_list[i]
            re.append(data)
        res.extend(re[0:20])
        random.seed(0)
        random.shuffle(re[20:2000])
        res.extend(re[20:])
        print(res[0], res[0]['pos'])
        print('processed %d files' % (len(res)))
    else:
        if dataset in ['qm7', 'qm8', 'qm9']:
            env = lmdb.open(
                os.path.join(lmdb_path, f"%s.lmdb" %dataset),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
            )
            txn = env.begin()
            keys = list(txn.cursor().iternext(values=False))
            for idx in keys:
                datapoint_pickled = txn.get(idx)
                qm_data.append(pickle.loads(datapoint_pickled))
            print(qm_data[0])
        else:
            qm_data = None
        res = []
        for i, f in enumerate(smiles_list):
            if i % 1000 == 0:
                print('processed %d smiles' %i)
            smiles = f
            if qm_data is not None:
                raw_qm_data = qm_data[i]
            else:
                raw_qm_data = None
            data = gen_geograph_data_from_smiles(raw_smiles=smiles, raw_qm_data=raw_qm_data, dataset=dataset)
            if data is None:
                continue
            if dataset in ['qm7', 'qm8', 'qm9']:
                data.y = torch.tensor(raw_qm_data['target'], dtype=torch.float32)
            else:
                data.y = target_list[i]
            res.append(data)
        print(res[0], res[0]['pos'])
        print('processed %d files' % (len(res)))
    return res

def idx2list(lis, idx):
    return [lis[_] for _ in idx]

def gen_GEOM_blocks(base_path, output_dir, dataset):

    smiles_list = []
    target_list = []

    smiles_path = os.path.join(base_path, "%s.csv" % dataset)
    with open(smiles_path, 'r') as f:
        reader = csv.reader(f)
        fields = next(reader)
        for line in reader:
            if not any(line):
                continue
            target = []
            for field, value in zip(fields, line):
                if field == "smiles":
                    smiles_list.append(value)
                elif dataset in ['bace', 'bbbp', 'sider', 'hiv']:
                    if value in ['0', 0, '0.0', 0.0]:
                        value = -1
                    if value in ['1.0', 1.0]:
                        value = 1
                    value = int(value)
                    target.append(value)
                elif dataset in ['clintox', 'tox21',  'toxcast', 'muv']:
                    if value in ['0', 0, '0.0', 0.0]:
                        value = -1
                    if value == "":
                        value = 0
                    if value in ['1.0', 1.0]:
                        value = 1
                    value = int(value)
                    target.append(value)
                elif dataset in ['esol', 'lipo', 'freesolv', 'qm7', 'qm8', 'qm9']:
                    value = float(value)
                    target.append(value)
                else:
                    raise ValueError('Dataset {} not included.'.format(dataset))

            target = torch.tensor(target)
            target_list.append(target)

    print('find %d smiles' % (len(smiles_list)))
    if len(target_list) != len(smiles_list):
        raise ValueError("Number of target `%s` doesn't match with number of molecules. "
                         "Expect %d but found %d" % (dataset, len(smiles_list), len(target_list)))

    print('start processing data')
    data = load_data_from_smiles(smiles_list, target_list, dataset)

    os.makedirs(os.path.join(base_path, '', output_dir), exist_ok=True)
    size = len(data)
    print(data[0])
    print('dataset_size: %d molecules.' % size)
    with open(os.path.join(base_path, '', output_dir, '%s_.pkl' % dataset), 'wb') as f:
        pickle.dump(data, f)
    with open(os.path.join(base_path, '', output_dir, '%s_summary_.json' % dataset), 'w') as f:
        json.dump({
            'dataset num': 1,
            'dataset size': size,
        }, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FinetuneData')
    parser.add_argument('--base_path', type=str, help='path of dataset', required=True)
    parser.add_argument('--output', type=str, help='output dir', required=True)
    parser.add_argument('--dataset', type=str, help='dataset for property prediction', required=True)
    args = parser.parse_args()

    gen_GEOM_blocks(args.base_path, args.output, args.dataset)