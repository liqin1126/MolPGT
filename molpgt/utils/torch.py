import copy
import warnings
import logging
import numpy as np

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from argparse import Namespace
from torch_sparse import SparseTensor

def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom

    
def repeat_data(data: Data, num_repeat) -> Batch:
    datas = [copy.deepcopy(data) for i in range(num_repeat)]
    return Batch.from_data_list(datas)



def repeat_batch(batch: Batch, num_repeat) -> Batch:
    datas = batch.to_data_list()
    new_data = []
    for i in range(num_repeat):
        new_data += copy.deepcopy(datas)
    return Batch.from_data_list(new_data)


#customize exp lr scheduler with min lr
class ExponentialLR_with_minLr(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, min_lr=1e-4, last_epoch=-1, verbose=False):
        self.gamma = gamma    
        self.min_lr = min_lr
        super(ExponentialLR_with_minLr, self).__init__(optimizer, gamma, last_epoch, verbose)    
        
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return self.base_lrs
        return [max(group['lr'] * self.gamma, self.min_lr)
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
                for base_lr in self.base_lrs]

class NoamLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch, init_lr, max_lr, final_lr):

        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs).astype(int)
        self.total_epochs = np.array(total_epochs).astype(int)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr).astype(float)
        self.max_lr = np.array(max_lr).astype(float)
        self.final_lr = np.array(final_lr).astype(float)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = (self.total_epochs * self.steps_per_epoch).astype(int)
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self):
        """Gets a list of the current learning rates."""
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
        If None, current_step = self.current_step + 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + int(self.current_step) * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]

def get_optimizer(config, model):
    if config.type == "Adam":
        return torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=config.lr, 
                        weight_decay=config.weight_decay)
    else:
        raise NotImplementedError('Optimizer not supported: %s' % config.type)


def get_scheduler(config, optimizer, train_size=None):
    if config.scheduler.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.factor,
            patience=config.patience,
            min_lr=float(config.min_lr),
        )
    elif config.scheduler.type == 'expmin':
        return ExponentialLR_with_minLr(
            optimizer,
            gamma=config.scheduler.factor,
            min_lr=float(config.scheduler.min_lr),
        )
    elif config.scheduler.type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            config.epochs, 
            eta_min=float(config.min_lr)
        )
    elif config.scheduler.type == 'noam':
        return NoamLR(
            optimizer,
            warmup_epochs=[config.scheduler.warmup_epochs],
            total_epochs=[config.epochs],
            steps_per_epoch=train_size // config.batch_size,
            init_lr=[config.lr],
            max_lr=[config.scheduler.max_lr],
            final_lr=[config.scheduler.final_lr]
        )
    else:
        raise NotImplementedError('Scheduler not supported: %s' % config.scheduler.type)

