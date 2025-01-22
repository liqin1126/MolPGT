import numpy as np
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean
import math
import time

loss_func = {
    "L1": nn.L1Loss(reduction='none'),
    "L2": nn.MSELoss(reduction='none'),
    "Cosine": nn.CosineSimilarity(dim=-1, eps=1e-08),
    "CrossEntropy": nn.CrossEntropyLoss()
}


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, config, rep_model, rep2_model):
        super().__init__()
        self.config = config
        self.hidden_dim = self.config.model.hidden_dim
        self.model = rep_model
        self.clmodel = rep2_model


    def forward(self, data):

        node2graph = data.batch
        pos = data.pos
        xl = self.model(data, pos, node2graph)
        clxl = self.clmodel(data, pos, node2graph)
        #xg = scatter_add(xl, node2graph, dim = -2)
        xg = xl.mean(1).view(xl.size(0), xl.size(-1))
        #clxg = scatter_add(clxl, node2graph, dim = -2)
        clxg =  clxl.mean(1).view(clxl.size(0), clxl.size(-1))
        # get contrastive loss
        batch_size = xg.size(0)
        emb = F.normalize(torch.cat([xg, clxg])).cuda()
        similarity = torch.matmul(emb, emb.t()).cuda() - torch.eye(batch_size * 2).cuda() * 1e12
        similarity = similarity * 20
        label = torch.tensor([(batch_size + i) % (batch_size * 2) for i in range(batch_size * 2)]).long().cuda()
        loss = loss_func["CrossEntropy"](similarity, label)

        return loss
