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
from sklearn.metrics import (accuracy_score, average_precision_score,
                             roc_auc_score)
from molpgt import utils
from molpgt.layers import Graph_Transformer
from molpgt.models import Classification
from molpgt.data import BatchDatapointProperty, GEOMDataset, balanced_scaffold_split, random_split, scaffold_split
from torch_geometric.loader import DataLoader

from collections import OrderedDict
import random
import yaml
from easydict import EasyDict
from time import time
import json

torch.multiprocessing.set_sharing_strategy('file_system')

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

    train_dataset = BatchDatapointProperty(train_dataset)
    train_dataset.load_datapoints()
    train_dataset = GEOMDataset([train_dataset], len(train_dataset), transforms=None)

    valid_dataset = BatchDatapointProperty(valid_dataset)
    valid_dataset.load_datapoints()
    valid_dataset = GEOMDataset([valid_dataset], len(valid_dataset), transforms=None)

    test_dataset = BatchDatapointProperty(test_dataset)
    test_dataset.load_datapoints()
    test_dataset = GEOMDataset([test_dataset], len(test_dataset), transforms=None)

    print(train_dataset[0], time()-start)

    # fix seed
    all_test_auroc = []
    all_final_test_auroc = []

    for i in range(config.train.num_run):
        train_start = time()
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

        # set up model
        def get_num_task(name):
            if name == 'tox21':
                return 12
            elif name in ['hiv', 'bace', 'bbbp']:
                return 1
            elif name == 'pcba':
                return 92
            elif name == 'muv':
                return 17
            elif name == 'toxcast':
                return 617
            elif name == 'sider':
                return 27
            elif name == 'clintox':
                return 2
            raise ValueError('Invalid dataset name.')

        # Bunch of classification tasks
        num_tasks = get_num_task(dataset_name)
        rep = Graph_Transformer(hidden_channels=config.model.hidden_dim, num_layers=config.model.n_layers,
                                num_heads=config.model.n_heads, dropout=config.model.dropout,
                                no_pos_encod=config.model.no_pos_encod, no_edge_update=config.model.no_edge_update)
        model = Classification(config=config, num_tasks=num_tasks,
                              model=rep)

        if config.train.restore_path:
            state = torch.load(config.train.restore_path, map_location=lambda storage, loc: storage)
            loaded_state_dict = state['model']
            model_state_dict = model.model.state_dict()

            # Skip missing parameters and parameters of mismatched size
            pretrained_state_dict = {}
            for param_name in loaded_state_dict.keys():
                if param_name[13:] not in model_state_dict:
                    continue
                elif model_state_dict[param_name[13:]].shape != loaded_state_dict[param_name].shape:
                    continue
                else:
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
        # different learning rates for different parts of model
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.train.lr,
                               weight_decay=float(config.train.weight_decay))
        scheduler = utils.get_scheduler(config.train, optimizer, len(train_dataset))
        eval_metric = roc_auc_score
        train_acc_list, val_roc_list, test_roc_list = [], [], []

        best_val_roc, best_val_idx = -1, 0

        for epoch in range(1, config.train.epochs + 1):
            #begin train
            epoch_time = time()
            if world_size > 1:
                train_sampler.set_epoch(epoch)
            model.train()
            total_loss = 0
            for step, batch in enumerate(train_loader):
                batch = batch.to(rank)
                pred = model(batch, batch.pos, batch.batch)
                if not pred.requires_grad:
                    raise RuntimeError("loss doesn't require grad")
                #y = batch.y.view(pred.shape).to(torch.float64)

                # Loss matrix
                y = batch.y.view(pred.shape).to(torch.float64)
                is_valid = y ** 2 > 0
                pred = pred[is_valid].float()
                y = y[is_valid].float()
                loss_mat = F.binary_cross_entropy_with_logits(
                    pred,
                    (y+1)/2,
                    reduction='none',
                )
                optimizer.zero_grad()
                loss = torch.sum(loss_mat) / torch.sum(is_valid)

                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.detach().item()
            loss_acc = total_loss / len(train_loader)

            if verbose:
                print('Epoch: {}\nTrain_Loss: {}\nTime: {}'.format(epoch, loss_acc, time()-epoch_time))

            if world_size > 1:
                dist.barrier()

            ##begin evaluation
            if world_size > 1:
                val_sampler.set_epoch(epoch)
            model.eval()
            epoch_val_time = time()
            y_true, y_scores = [], []

            for step, batch in enumerate(val_loader):
                batch = batch.to(rank)
                with torch.no_grad():
                    pred = model(batch, batch.pos, batch.batch)

                probs =torch.sigmoid(pred.float()).view(-1, pred.size(-1))
                true = batch.y.view(pred.shape)

                y_true.append(true)
                y_scores.append(probs)

            y_true = torch.cat(y_true, dim=0).cpu().numpy()
            y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

            roc_list = []
            if dataset_name in ['bbbp', 'bace', 'hiv']:
                if np.sum(y_true == 1) > 0 and np.sum(y_true == -1) > 0:
                    val_roc = eval_metric((y_true + 1) / 2, y_scores)
                else:
                    continue
            else:
                for i in range(y_true.shape[1]):
                    # AUC is only defined when there is at least one positive data.
                    if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
                        is_valid = y_true[:, i] ** 2 > 0
                        roc_list.append(eval_metric((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
                    else:
                        #if rank == 0:
                        #    print('{} is invalid'.format(i))
                        continue
                if len(roc_list) < y_true.shape[1]:
                    if rank == 0:
                        print(len(roc_list))
                        #print('Some target is missing!')
                        print('Missing ratio: %f' % (1 - float(len(roc_list)) / y_true.shape[1]))
                val_roc = sum(roc_list) / len(roc_list)
            val_target = y_true
            val_pred = y_scores

            ##begin test
            if world_size > 1:
                test_sampler.set_epoch(epoch)
            model.eval()
            y_true, y_scores = [], []

            for step, batch in enumerate(test_loader):
                batch = batch.to(rank)
                with torch.no_grad():
                    pred = model(batch, batch.pos, batch.batch)

                probs =torch.sigmoid(pred.float()).view(-1, pred.size(-1))
                true = batch.y.view(pred.shape)

                y_true.append(true)
                y_scores.append(probs)

            y_true = torch.cat(y_true, dim=0).cpu().numpy()
            y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

            roc_list = []
            if dataset_name in ['bbbp', 'bace', 'hiv']:
                if np.sum(y_true == 1) > 0 and np.sum(y_true == -1) > 0:
                    test_roc = eval_metric((y_true + 1) / 2, y_scores)
                else:
                    continue
            else:
                for i in range(y_true.shape[1]):
                    # AUC is only defined when there is at least one positive data.
                    if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
                        is_valid = y_true[:, i] ** 2 > 0
                        roc_list.append(eval_metric((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
                    else:
                        #if rank == 0:
                        #    print('{} is invalid'.format(i))
                        continue
                if len(roc_list) < y_true.shape[1]:
                    if rank == 0:
                        print(len(roc_list))
                        #print('Some target is missing!')
                        print('Missing ratio: %f' % (1 - float(len(roc_list)) / y_true.shape[1]))
                test_roc = sum(roc_list) / len(roc_list)
            test_target = y_true
            test_pred = y_scores

            train_acc_list.append(loss_acc)
            val_roc_list.append(val_roc)
            test_roc_list.append(test_roc)
            if rank==0:
                print('val: {:.6f}\ttest: {:.6f}\ttime:{:.2f}'.format(val_roc, test_roc, time()-epoch_val_time))

            if not val_roc < best_val_roc:
                best_val_roc = val_roc
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
            print(
                'best epoch:{}\ttrain: {:.6f}\tval: {:.6f}\ttest: {:.6f}\tlr: {}'.format(best_val_idx+1,
                    train_acc_list[best_val_idx], val_roc_list[best_val_idx],
                                   test_roc_list[best_val_idx], optimizer.param_groups[0]['lr']))

        if not config.train.save_path == '':
            output_model_path = os.path.join(config.train.save_path, config.model.name, 'run%s' % run_seed,
                                             'model_final.pth')
            saved_model_dict = {
                'model': model.state_dict()
            }
            torch.save(saved_model_dict, output_model_path)

        all_test_auroc.append(test_roc_list[best_val_idx])
        all_final_test_auroc.append(test_roc_list[epoch-1])

    all_test_auroc = np.array(all_test_auroc)
    all_final_test_auroc = np.array(all_final_test_auroc)
    if rank == 0:
        print('all_test_auroc: {}\tmean: {:.6f}\tstd: {:.6f}'.format(all_test_auroc, np.nanmean(all_test_auroc),
                                                                     np.nanstd(all_test_auroc)))
        print('all_final_test_auroc: {}\tmean: {:.6f}\tstd: {:.6f}'.format(all_final_test_auroc, np.nanmean(all_final_test_auroc),
                                                                     np.nanstd(all_final_test_auroc)))
        print('dataset: {}\tbsz: {}'.format(config.model.name, config.train.batch_size))
    if world_size > 1:
        dist.destroy_process_group()



if __name__ == '__main__':

    main()
