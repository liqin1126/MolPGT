import copy
import pandas as pd
import math
import os
import sys
sys.path.append('.')

import numpy as np
import pickle
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

import argparse

from sklearn.metrics import mean_absolute_error, mean_squared_error
from emegt import utils
from emegt.layers import Graph_Transformer
from emegt.models import MolProperty
from emegt.data import BatchDatapointProperty, GEOMDataset, balanced_scaffold_split, random_split, scaffold_split
from torch_geometric.loader import DataLoader

from collections import OrderedDict
import random
import yaml
from easydict import EasyDict
from time import time
import json

torch.multiprocessing.set_sharing_strategy('file_system')
task_metainfo = {
    "esol": {
        "mean": -3.0501019503546094,
        "std": 2.096441210089345,
        "target_name": "logSolubility",
    },
    "freesolv": {
        "mean": -3.8030062305295944,
        "std": 3.8478201171088138,
        "target_name": "freesolv",
    },
    "lipo": {"mean": 2.186336, "std": 1.203004, "target_name": "lipo"},
    "qm7": {
        "mean": -1544.8360893118609,
        "std": 222.8902092792289,
        "target_name": "u0_atom",
    },
    "qm8": {
        "mean": [
            0.22008500524052105,
            0.24892658759891675,
            0.02289283121913152,
            0.043164444107224746,
            0.21669716560818883,
            0.24225989336408812,
            0.020287111373358993,
            0.03312609817084387,
            0.21681478862847584,
            0.24463634931699113,
            0.02345177178004201,
            0.03730141834205415,
        ],
        "std": [
            0.043832862248693226,
            0.03452326954549232,
            0.053401140662012285,
            0.0730556474716259,
            0.04788020599385645,
            0.040309670766319,
            0.05117163534626215,
            0.06030064428723054,
            0.04458294838213221,
            0.03597696243350195,
            0.05786865052149905,
            0.06692733477994665,
        ],
        "target_name": [
            "E1-CC2",
            "E2-CC2",
            "f1-CC2",
            "f2-CC2",
            "E1-PBE0",
            "E2-PBE0",
            "f1-PBE0",
            "f2-PBE0",
            "E1-CAM",
            "E2-CAM",
            "f1-CAM",
            "f2-CAM",
        ],
    },
    "qm9": {
        "mean": [-0.23997669940621352, 0.011123767412331285, 0.2511003712141015],
        "std": [0.02213143402267657, 0.046936069870866196, 0.04751888787058615],
        "target_name": ["homo", "lumo", "gap"],
    },
}

def main():

    torch.set_printoptions(profile="full")

    parser = argparse.ArgumentParser(description='molecule property prediction')
    parser.add_argument('--config_path', type=str, default='.', metavar='N',
                        help='Path of config yaml.')
    parser.add_argument('--model_name', type=str, default='', metavar='N',
                        help='Model name.')
    parser.add_argument('--restore_path', type=str, default='', metavar='N',
                        help='Restore path.')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()

    #device = torch.device("cuda")

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    if args.model_name != '':
        config.model.name = args.model_name

    if args.restore_path != '':
        config.train.restore_path = args.restore_path

    print(config)

    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    rank = args.local_rank
    verbose=1
    if world_size > 1:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
    print('Rank: ',rank)
    if rank != 0:
        verbose = 0

    start = time()

    data_dir = config.data.data_dir
    dataset_name = config.model.name
    with open(os.path.join(data_dir, '%s_summary.json' % dataset_name), 'r') as f:
        summ = json.load(f)
    dataset_size = summ['dataset size']
    with open(os.path.join(data_dir, '%s.pkl' % dataset_name), 'rb') as f:
        datas = pickle.load(f)
    assert dataset_size == len(datas)

    if config.data.split == 'scaffold':
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            datas, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1)
        print('split via scaffold')
    elif config.data.split == 'random':
        train_dataset, valid_dataset, test_dataset = random_split(
            datas, null_value=0, frac_train=0.8, frac_valid=0.1,
            frac_test=0.1)
        print('randomly split')
    elif config.data.split == 'balanced_scaffold':
        train_dataset, valid_dataset, test_dataset = balanced_scaffold_split(
            datas, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1)
        print('balanced scaffold')
    else:
        raise ValueError('Invalid split option.')

    '''
    if config.model.name in ['qm7', 'qm8', 'qm9']:
        size = 11
        for i in range(len(train_dataset)):
            seed = int(hash((config.train.seed, i)) % 1e6)
            state = np.random.get_state()
            np.random.seed(seed)
            np.random.set_state(state)
            sample_idx = np.random.randint(size)
            train_dataset[i]['pos'] = train_dataset[i]['pos'].view(11, -1, 3)[sample_idx]
        val_len = len(valid_dataset) * 11
        val_data = [None] * val_len
        for i in range(val_len):
            mol_idx = i // size
            pos_idx = i % size
            pos = valid_dataset[mol_idx]['pos'].view(11, -1, 3)[pos_idx]
            val_data[i] = copy.deepcopy(valid_dataset[mol_idx])
            val_data[i]['pos'] = copy.deepcopy(pos)
        test_len = len(test_dataset) * 11
        test_data = [None] * test_len
        for i in range(test_len):
            mol_idx = i // size
            pos_idx = i % size
            pos = test_dataset[mol_idx]['pos'].view(11, -1, 3)[pos_idx]
            test_data[i] = copy.deepcopy(test_dataset[mol_idx])
            test_data[i]['pos'] = copy.deepcopy(pos)
            
        valid_dataset = BatchDatapointProperty(val_data)
        valid_dataset.load_datapoints()
        valid_dataset = GEOMDataset([val_data], val_len, transforms=None)

        test_dataset = BatchDatapointProperty(test_data)
        test_dataset.load_datapoints()
        test_dataset = GEOMDataset([test_data], test_len, transforms=None)
    
    if config.model.name in ['qm7', 'qm8', 'qm9']:
        size = 11
        for i in range(len(train_dataset)):
            seed = int(hash((config.train.seed, i)) % 1e6)
            state = np.random.get_state()
            np.random.seed(seed)
            np.random.set_state(state)
            sample_idx = np.random.randint(size)
            train_dataset[i]['pos'] = train_dataset[i]['pos'].view(11, -1, 3)[sample_idx]
        for i in range(len(valid_dataset)):
            seed = int(hash((config.train.seed, len(train_dataset)+i)) % 1e6)
            state = np.random.get_state()
            np.random.seed(seed)
            np.random.set_state(state)
            sample_idx = np.random.randint(size)
            valid_dataset[i]['pos'] = valid_dataset[i]['pos'].view(11, -1, 3)[sample_idx]
        for i in range(len(test_dataset)):
            seed = int(hash((config.train.seed, len(train_dataset)+len(valid_dataset)+i)) % 1e6)
            state = np.random.get_state()
            np.random.seed(seed)
            np.random.set_state(state)
            sample_idx = np.random.randint(size)
            test_dataset[i]['pos'] = test_dataset[i]['pos'].view(11, -1, 3)[sample_idx]
    '''

    train_dataset = BatchDatapointProperty(train_dataset)
    train_dataset.load_datapoints()
    train_dataset = GEOMDataset([train_dataset], len(train_dataset), transforms=None)

    valid_dataset = BatchDatapointProperty(valid_dataset)
    valid_dataset.load_datapoints()
    valid_dataset = GEOMDataset([valid_dataset], len(valid_dataset), transforms=None)

    test_dataset = BatchDatapointProperty(test_dataset)
    test_dataset.load_datapoints()
    test_dataset = GEOMDataset([test_dataset], len(test_dataset), transforms=None)

    print(train_dataset[0], valid_dataset[0], test_dataset[0])

    print(time() - start)

    # fix seed
    all_test_result = []
    all_final_test_result = []

    for i in range(config.train.num_run):
        print(f'Run {i}')
        run_seed = config.train.seed + i
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(run_seed)
            torch.cuda.manual_seed_all(run_seed)
        torch.backends.cudnn.benchmark = True
        print('fix seed:', run_seed)

        os.makedirs(config.train.save_path + "/" + config.model.name + "/" + 'run%s' % run_seed, exist_ok=True)


        if world_size == 1:
            train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size,
                                      shuffle=True, num_workers=config.train.num_workers, pin_memory=False)
            val_loader = DataLoader(valid_dataset, batch_size=config.train.batch_size,
                                    shuffle=False, num_workers=config.train.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=config.train.batch_size,
                                     shuffle=False, num_workers=config.train.num_workers)

        else:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
            train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, sampler=train_sampler,
                                      num_workers=config.train.num_workers, pin_memory=False)
            val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)
            val_loader = DataLoader(valid_dataset, batch_size=config.train.batch_size, sampler=val_sampler,
                                      num_workers=config.train.num_workers)
            test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
            test_loader = DataLoader(test_dataset, batch_size=config.train.batch_size, sampler=test_sampler,
                                      num_workers=config.train.num_workers)

        def get_num_task(name):
            if name in ['esol', 'lipo', 'freesolv', 'qm7']:
                return 1
            elif name == 'qm8':
                return 12
            elif name == 'qm9':
                return 3
            raise ValueError('Invalid dataset name.')

        # Bunch of classification tasks
        num_tasks = get_num_task(dataset_name)
        rep = Graph_Transformer(hidden_channels=config.model.hidden_dim, num_layers=config.model.n_layers,
                                num_heads=config.model.n_heads, dropout=config.model.dropout,
                                no_pos_encod=config.model.no_pos_encod, no_edge_update=config.model.no_edge_update)
        model = MolProperty(config=config, num_tasks=num_tasks,
                              model=rep)


        if config.train.restore_path:
            state = torch.load(config.train.restore_path, map_location=lambda storage, loc: storage)
            loaded_state_dict = state['model']
            model_state_dict = model.model.state_dict()

            # Skip missing parameters and parameters of mismatched size
            pretrained_state_dict = {}
            for param_name in loaded_state_dict.keys():
                if param_name[13:] not in model_state_dict:
                    #print(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
                    continue
                elif model_state_dict[param_name[13:]].shape != loaded_state_dict[param_name].shape:
                    #print(f'Pretrained parameter "{param_name}" '
                    #      f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                    #      f'model parameter of shape {model_state_dict[param_name[13:]].shape}.')
                    continue
                else:
                    # debug(f'Loading pretrained parameter "{param_name}".')
                    #print(param_name)
                    pretrained_state_dict[param_name[13:]] = loaded_state_dict[param_name]

            # Load pretrained weights
            model_state_dict.update(pretrained_state_dict)
            model.model.load_state_dict(model_state_dict)
            print('load model from', config.train.restore_path)

        model=model.to(rank)
        if world_size > 1:
            model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
        print(model)
        print(sum(p.numel() for p in model.parameters()))

        # set up optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.train.lr,
                               weight_decay=float(config.train.weight_decay))
        scheduler = utils.get_scheduler(config.train, optimizer, len(train_dataset))

        train_result_list, val_result_list, test_result_list = [], [], []
        best_val_metric, best_val_idx = 1e10, 0

        for epoch in range(1, config.train.epochs + 1):
            #begin train
            epoch_time = time()
            if world_size > 1:
                train_sampler.set_epoch(epoch)
            model.train()
            total_loss = 0
            for step, batch in enumerate(train_loader):
                batch = batch.to(rank)
                pred = model(batch, batch.pos, batch.batch).view(-1, num_tasks).float()
                if not pred.requires_grad:
                    raise RuntimeError("loss doesn't require grad")
                y = batch.y.view(-1, num_tasks).float()
                y_mean = torch.tensor(task_metainfo[dataset_name]['mean'], device=y.device)
                y_std = torch.tensor(task_metainfo[dataset_name]['std'], device=y.device)
                y = (y - y_mean) / y_std
                if dataset_name in ['esol', 'lipo', 'freesolv']:
                    loss = F.mse_loss(pred, y, reduction='sum')
                else:
                    loss = F.smooth_l1_loss(pred, y, reduction="sum")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += (loss/len(y)).detach().item()

            loss_acc = total_loss / len(train_loader) /math.log(2)

            if verbose:
                print('Epoch: {}\nTrain_Loss: {}\nTime: {}'.format(epoch, loss_acc, time()-epoch_time))

            if world_size > 1:
                dist.barrier()

            ##begin evaluation
            if world_size > 1:
                val_sampler.set_epoch(epoch)
            model.eval()
            epoch_val_time = time()
            y_true, y_pred = [], []
            for step, batch in enumerate(val_loader):
                batch = batch.to(rank)
                y_mean = torch.tensor(task_metainfo[dataset_name]['mean'], device=batch.y.device)
                y_std = torch.tensor(task_metainfo[dataset_name]['std'], device=batch.y.device)
                with torch.no_grad():
                    pred = model(batch, batch.pos, batch.batch).squeeze(1)
                    pred = pred * y_std + y_mean

                true = batch.y.view(pred.shape)

                y_true.append(true)
                y_pred.append(pred)

            if dataset_name in ['esol', 'lipo', 'freesolv']:
                y_true = torch.cat(y_true, dim=0).cpu().numpy()
                y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
                rmse = mean_squared_error(y_true, y_pred, squared=False)
                val_result = rmse
            else:
                #y_true = torch.cat(y_true, dim=0).view(-1, 11, num_tasks).cpu().numpy().mean(axis=1)
                #y_pred = torch.cat(y_pred, dim=0).view(-1, 11, num_tasks).cpu().numpy().mean(axis=1)
                y_true = torch.cat(y_true, dim=0).cpu().numpy()
                y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
                val_result = np.abs(y_pred - y_true).mean()

            val_target, val_pred = y_true, y_pred

            ##begin test
            if world_size > 1:
                test_sampler.set_epoch(epoch)
            model.eval()
            y_true, y_pred = [], []
            for step, batch in enumerate(test_loader):
                batch = batch.to(rank)
                y_mean = torch.tensor(task_metainfo[dataset_name]['mean'], device=batch.y.device)
                y_std = torch.tensor(task_metainfo[dataset_name]['std'], device=batch.y.device)
                with torch.no_grad():
                    pred = model(batch, batch.pos, batch.batch).squeeze(1)
                    pred = pred * y_std + y_mean

                true = batch.y.view(pred.shape)

                y_true.append(true)
                y_pred.append(pred)

            if dataset_name in ['esol', 'lipo', 'freesolv']:
                y_true = torch.cat(y_true, dim=0).cpu().numpy()
                y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
                rmse = mean_squared_error(y_true, y_pred, squared=False)
                test_result = rmse
            else:
                #y_true = torch.cat(y_true, dim=0).view(-1, 11, num_tasks).cpu().numpy().mean(axis=1)
                #y_pred = torch.cat(y_pred, dim=0).view(-1, 11, num_tasks).cpu().numpy().mean(axis=1)
                y_true = torch.cat(y_true, dim=0).cpu().numpy()
                y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
                test_result = np.abs(y_pred - y_true).mean()

            test_target, test_pred = y_true, y_pred

            train_result_list.append(loss_acc)
            val_result_list.append(val_result)
            test_result_list.append(test_result)

            if rank == 0:
                if dataset_name in ['esol', 'lipo', 'freesolv']:
                    print('RMSE val: {:.6f}\ttest: {:.6f}\ttime:{:.2f}'.format(val_result,
                                                                         test_result, time()-epoch_val_time))
                else:
                    print('MAE val: {:.6f}\ttest: {:.6f}\ttime:{:.2f}'.format(val_result,
                                                                         test_result, time()-epoch_val_time))
            if not val_result > best_val_metric:
                best_val_metric = val_result
                best_val_idx = epoch - 1
                if not config.train.save_path == '':
                    output_model_path = os.path.join(config.train.save_path, config.model.name, 'run%s' %run_seed,
                                                     'model_best.pth')
                    saved_model_dict = {
                        'model': model.state_dict()
                    }
                    torch.save(saved_model_dict, output_model_path)

                    filename = os.path.join(config.train.save_path, config.model.name, 'run%s' %run_seed,
                                            'evaluation_best.pth')
                    np.savez(filename, val_target=val_target, val_pred=val_pred,
                             test_target=test_target, test_pred=test_pred)

            if world_size > 1:
                dist.barrier()

        if rank == 0:
            print('Best epoch: {}\ttrain: {:.6f}\tval: {:.6f}\ttest: {:.6f}\tlr: {}'.format(
                best_val_idx+1, train_result_list[best_val_idx], val_result_list[best_val_idx],
                test_result_list[best_val_idx], optimizer.param_groups[0]['lr']))

        if not config.train.save_path == '':
            output_model_path = os.path.join(config.train.save_path, config.model.name, 'run%s' % run_seed,
                                             'model_final.pth')
            saved_model_dict = {
                'model': model.state_dict()
            }
            torch.save(saved_model_dict, output_model_path)

        all_test_result.append(test_result_list[best_val_idx])
        all_final_test_result.append(test_result_list[epoch-1])


    all_test_result = np.array(all_test_result)
    all_final_test_result = np.array(all_final_test_result)
    if rank == 0:
        print('all_test_result: {}\tmean: {:.6f}\tstd: {:.6f}'.format(all_test_result, np.nanmean(all_test_result),
                                                                     np.nanstd(all_test_result)))
        print('all_final_test_result: {}\tmean: {:.6f}\tstd: {:.6f}'.format(all_final_test_result, np.nanmean(all_final_test_result),
                                                                     np.nanstd(all_final_test_result)))
        print('dataset: {}\tbsz: {}'.format(config.model.name, config.train.batch_size))


    if world_size > 1:
        dist.destroy_process_group()



if __name__ == '__main__':

    main()
