#coding: utf-8

import argparse
import numpy as np
import random
import os
import yaml
from easydict import EasyDict

import torch
import sys
sys.path.append('.')
from torch_geometric.loader import DataLoader
from emegt import utils, layers, models
from time import time

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from emegt.data import BatchDatapoint, GEOMDataset
import json

torch.multiprocessing.set_sharing_strategy('file_system')


def train(rank, config, world_size, verbose=1):

    print('Rank: ', rank)
    if rank != 0:
        verbose = 0

    train_start = time()

     # set random seed
    np.random.seed(config.train.seed)
    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.train.seed)        
        torch.cuda.manual_seed_all(config.train.seed)
    torch.backends.cudnn.benchmark = True

    data_dir = config.data.block_dir

    with open(os.path.join(data_dir,'summary.json'),'r') as f:
        summ = json.load(f)
    
    train_block_size = summ['train block size']
    val_block_size = summ['val block size']
    val_block = BatchDatapoint(os.path.join(data_dir, 'val_block.pkl'), val_block_size)
    val_block.load_datapoints()
    val_dataset = GEOMDataset([val_block], val_block_size, transforms=None)
    train_block = BatchDatapoint(os.path.join(data_dir, 'train_block.pkl'), train_block_size)
    train_block.load_datapoints()
    train_dataset = GEOMDataset([train_block], train_block_size, transforms=None)

    rep = layers.Graph_Transformer(hidden_channels=config.model.hidden_dim, num_layers=config.model.n_layers,
                                   num_heads=config.model.n_heads, dropout=config.model.dropout,
                                   no_pos_encod=config.model.no_pos_encod,
                                   no_edge_update=config.model.no_edge_update)

    print('Number of M1 parameters =', sum(param.numel() for param in rep.parameters() if param.requires_grad))
    rep2 = layers.Graph_Transformer(hidden_channels=config.model.hidden_dim, num_layers=2,
                                    num_heads=config.model.n_heads, dropout=config.model.dropout,
                                    no_pos_encod=config.model.no_pos_encod, no_edge_update=config.model.no_edge_update, contrastive=True)
    print('Number of M2 parameters =', sum(param.numel() for param in rep2.parameters() if param.requires_grad))
    model = models.ContrastiveLoss(config, rep, rep2)

    num_epochs = config.train.epochs
    if world_size == 1:
        dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True,
                            num_workers=config.train.num_workers, pin_memory=False)
    else:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                        rank=rank)
        dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size,
                                sampler=train_sampler, num_workers=config.train.num_workers, pin_memory=False)

    valloader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False,
                           num_workers=config.train.num_workers)
  
    model = model.to(rank)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = utils.get_optimizer(config.train.optimizer, model)
    scheduler = utils.get_scheduler(config.train, optimizer)
    train_losses = []
    val_losses = []
    ckpt_list = []
    max_ckpt_maintain = 10
    best_loss = 100.0
    start_epoch = 0
    
    print(f'Rank {rank} start training...')
    
    for epoch in range(num_epochs):
        #train
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_start = time()
        batch_losses = []
        batch_cnt = 0
        for batch in dataloader:
            batch_cnt += 1
            batch = batch.to(rank)
            loss = model(batch)
            if not loss.requires_grad:
                raise RuntimeError("loss doesn't require grad")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

            if verbose and (batch_cnt % config.train.log_interval == 0 or (epoch==0 and batch_cnt <= 10)):
                print('Epoch: %d | Step: %d | loss: %.6f(%.3f)| Lr: %.5f' % \
                                    (epoch + start_epoch, batch_cnt, batch_losses[-1], loss.item(), optimizer.param_groups[0]['lr']))


        average_loss = sum(batch_losses) / (len(batch_losses) * config.train.batch_size)
        train_losses.append(average_loss)

        if verbose:
            print('Epoch: %d | Train Loss: %.6f | Time: %.5f' % (epoch + start_epoch, average_loss, time() - epoch_start))
        
        scheduler.step()

        if world_size > 1:
            dist.barrier()

        model.eval()
        eval_start = time()
        eval_losses = []
        for batch in valloader:
            batch = batch.to(rank)  
            loss = model(batch)
            eval_losses.append(loss.item())       
        average_loss = sum(eval_losses) / (len(eval_losses) * config.train.batch_size)

        if rank == 0:

            print('Evaluate val Loss: %.6f | Time: %.5f' % (average_loss, time() - eval_start))
            
            val_losses.append(average_loss)

            if val_losses[-1] < best_loss:
                best_loss = val_losses[-1]
                if config.train.save:
                    state = {
                        "model": model.state_dict(),
                        "config": config,
                        'cur_epoch': epoch + start_epoch,
                        'best_loss': best_loss,
                    }
                    epoch = str(epoch) if epoch is not None else ''
                    checkpoint = os.path.join(config.train.save_path,'checkpoint%s' % epoch)

                    if len(ckpt_list) >= max_ckpt_maintain:
                        try:
                            os.remove(ckpt_list[0])
                        except:
                            print('Remove checkpoint failed for', ckpt_list[0])
                        ckpt_list = ckpt_list[1:]
                        ckpt_list.append(checkpoint)
                    else:
                        ckpt_list.append(checkpoint)

                    torch.save(state, checkpoint)
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        state = {
            "model": model.module.state_dict()
        }
        checkpoint = os.path.join(config.train.save_path, 'checkpoint_final')
        torch.save(state, checkpoint)
        print('optimization finished.')
        print('Total time elapsed: %.5fs' % (time() - train_start))
    if world_size > 1:
        dist.destroy_process_group()

def main():

    torch.set_printoptions(profile="full")

    parser = argparse.ArgumentParser(description='contrastive')
    parser.add_argument('--config_path', type=str, help='path of config yaml', required=True)
    parser.add_argument('--seed', type=int, default=0, help='overwrite config seed')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    if args.seed != 0:
        config.train.seed = args.seed

    if config.train.save and config.train.save_path is not None:
        config.train.save_path = os.path.join(config.train.save_path, config.model.name)
        if not os.path.exists(config.train.save_path):
            os.makedirs(config.train.save_path, exist_ok=True)

    print(config)

    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')

    if world_size > 1:
        dist.init_process_group('nccl', rank=args.local_rank, world_size=world_size)
    train(args.local_rank, config, world_size)

if __name__ == '__main__':
    main()


