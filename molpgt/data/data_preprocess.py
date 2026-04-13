import os
import pickle
import copy
import json
import csv
from collections import defaultdict
from rdkit.Chem import AllChem
import numpy as np
import random


import sys
sys.path.append('qm9')
from torch_geometric.data import Data, Dataset

from rdkit import RDLogger
from graph import mol_to_geograph_data_MMFF3d
RDLogger.DisableLog('rdApp.*')
import argparse
from multiprocessing import Pool, set_start_method
set_start_method("spawn", force=True)

def gen_geograph_data_from_smiles(raw_smiles):
    smiles = raw_smiles[0]
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None
    data = mol_to_geograph_data_MMFF3d(mol, smiles=smiles)
    data['smiles'] = smiles
    return data

def load_data_from_smiles(smiles_list, worker_id):

    res = []
    for i, f in enumerate(smiles_list):
        if i % 10000 == 0:
            print('worker %d, processed %d smiles' % (worker_id, i))

        smiles = f
        data = gen_geograph_data_from_smiles(raw_smiles=smiles)
        res.append(data)
    print('worker %d, processed %d files' % (worker_id, len(smiles_list)))
    return res

def idx2list(lis, idx):
    return [lis[_] for _ in idx]

def gen_train_val(base_path, val_num = 200, workers = 10, seed=None):
    # set random seed
    if seed is None:
        seed = 2023
    np.random.seed(seed)
    random.seed(seed)
    smiles_list = []

    # read zinc_250k_smiles file
    smiles_path = os.path.join(base_path, "raw", "zinc15_250K.csv")
    with open(smiles_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            smiles_list.append(line)
    print('find %d smiles' % (len(smiles_list)))

    split_indexes = list(range(len(smiles_list)))
    random.shuffle(split_indexes)
    val = split_indexes[:val_num]
    print(len(val))
    train = split_indexes[val_num:]

    if workers == 1:
        print('start processing val data')
        val_data = load_data_from_smiles(idx2list(smiles_list, val), 0)
        print('start processing train data')
        train_data = load_data_from_smiles(idx2list(smiles_list, train), 0)
    else:
        print('start processing val data')
        val_p = Pool(processes=workers)
        val_f_per_w = int(np.ceil(len(val) / workers))
        val_wks = []
        for i in range(workers):
            val_idx = val[i * val_f_per_w : (i+1) * val_f_per_w]
            wk = val_p.apply_async(
                load_data_from_smiles, (idx2list(smiles_list, val_idx), i)
            )
            val_wks.append(wk)
        
        val_p.close()
        val_p.join()
        val_data = []
        for wk in val_wks:
            _val_data, = wk.get()
            val_data.extend(_val_data)

        print('start processing train data')
        train_p = Pool(processes=workers)
        train_f_per_w = int(np.ceil(len(train) / workers))
        train_wks = []
        for i in range(workers):
            tar_idx = train[i * train_f_per_w : (i+1) * train_f_per_w]
            wk = train_p.apply_async(
                load_data_from_smiles, (idx2list(smiles_list, tar_idx), i)
            )
            train_wks.append(wk)
        
        train_p.close()
        train_p.join()  
        train_data = []
        for wk in train_wks:
            _train_data = wk.get()
            train_data.extend(_train_data)

    print('train size: %d molecules.' % (len(train_data)))
    print('val size: %d molecules.' % (len(val_data)))

    return train_data, val_data

def gen_GEOM_blocks(base_path, output_dir, val_num = 200, workers = 10, seed=None):
    train_data, val_data = gen_train_val(base_path, val_num, workers, seed)
    os.makedirs(os.path.join(base_path, '', output_dir), exist_ok=True)
    train_size = len(train_data)
    with open(os.path.join(base_path, '', output_dir, 'train_block.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    val_size = len(val_data)
    with open(os.path.join(base_path, '', output_dir, 'val_block.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
    with open(os.path.join(base_path, '', output_dir, 'summary.json'), 'w') as f:
        json.dump({
            'train block num': 1,
            'train block size': train_size,
            'val block num': 1,
            'val block size': val_size
        }, f)

example = {'charge','ensembleenergy','ensembleentropy','ensemblefreeenergy','lowestenergy', 'poplowestpct', 'temperature', 'totalconfs', 'uniqueconfs'}

def gen_summary(base_path, pkl_dir):
    files = os.listdir(os.path.join(base_path, pkl_dir))
    tar_dic = {}
    for f in files:
        tk = '.'.join(f.split('.')[:-1])
        tv = {}
        with open(os.path.join(base_path,pkl_dir,f),'rb') as fin:
            p = pickle.load(fin)
        for k,v in p.items():
            if k in example:
                tv[k] = v
        tv['pickle_path'] = os.path.join(pkl_dir, f)
        tar_dic[tk] = tv
    with open(os.path.join(base_path, 'summary_%s.json' % pkl_dir), 'w') as f:
        json.dump(tar_dic, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GeomData')
    parser.add_argument('--base_path', type=str, help='path of dataset', required=True)
    parser.add_argument('--output', type=str, help='output dir', required=True)
    parser.add_argument('--val_num', type=int, help='maximum moleculars for validation', default = 1000)
    parser.add_argument('--num_workers', type=int, help='workers number', default = 1)
    args = parser.parse_args()

    gen_GEOM_blocks(args.base_path, args.output, val_num=args.val_num, workers = args.num_workers)