from torch import nn
import pickle
from .embedding import AtomEmbedding, AtomFloatEmbedding, BondEmbedding, BondFloatRBF
from .rt import Basic_RT

import torch
import numpy as np

ele2emb = pickle.load(open('kgembedding/ele2emb.pkl','rb'))

class Graph_Transformer(nn.Module):
    r"""The TorchMD equivariant Transformer architecture.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_layers (int, optional): The number of attention layers.
            (default: :obj:`6`)
        num_heads (int, optional): Number of attention heads.
            (default: :obj:`8`)
    """

    def __init__(
        self,
        hidden_channels=128,
        num_layers=6,
        num_heads=8,
        dropout=0.0,
        no_pos_encod=False,
        no_edge_update=False,
        contrastive=False,
    ):
        super(Graph_Transformer, self).__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.contrastive = contrastive
        self.no_pos_encod = no_pos_encod
        self.no_edge_update = no_edge_update

        self.atom_names = [
            "atom_type", "chiral_tag", "degree", "explicit_valence",
            "formal_charge", "hybridization", "implicit_valence",
            "is_aromatic", "total_numHs"
        ]
        self.bond_names = ["edge_type", "bond_dir", "is_in_ring"]

        self.init_atom_embedding = AtomEmbedding(self.atom_names, self.hidden_channels)
        self.init_atom_float_embedding = AtomFloatEmbedding(["mass"], self.hidden_channels)
        self.init_bond_embedding = BondEmbedding(self.bond_names, self.hidden_channels)
        self.init_bond_float_embedding = BondFloatRBF(["bond_length"], self.hidden_channels)
        self.init_function_embedding = nn.Linear(133, self.hidden_channels)

        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attention_layers.append(Basic_RT(self.num_heads, self.hidden_channels, self.dropout,
                                                  self.no_edge_update))


    def forward(self, data, pos, node2graph):

        e_emb = self.init_bond_embedding(data)
        device = data.pos.device
        if not self.contrastive:
            n_tensors = self.init_atom_embedding(data) + self.init_atom_float_embedding(data)
        else:
            f_tensors = []
            atomic_num = data.atom_type.tolist()
            for i in atomic_num:
                f_tensors.append(ele2emb[i].tolist())
            f_tensors = torch.tensor(f_tensors).to(device)
            n_tensors = self.init_function_embedding(f_tensors)
        e2g = []
        e_index_row = []
        e_index_col = []
        n_num = []
        for i in range(data.num_graphs):
            i_index = torch.nonzero(torch.eq(node2graph, i)).view(-1).tolist()
            i_num = len(i_index)
            n_num.append(i_num)
            e2g += torch.full((i_num*i_num,), i).tolist()
            e_index_col = e_index_col + i_index * i_num
            for j in i_index:
                e_index_row += torch.full((i_num,), j).tolist()
        e2g = torch.tensor(e2g).to(device)
        e_index_row = torch.tensor(e_index_row).unsqueeze(0).to(device)
        e_index_col = torch.tensor(e_index_col).unsqueeze(0).to(device)
        e_index = torch.cat((e_index_row, e_index_col), dim=0)
        if not self.no_pos_encod:
            dist = (pos[e_index_row] - pos[e_index_col]).norm(dim=-1).to(device)
            e_tensors = self.init_bond_float_embedding(dist)
        else:
            e_tensors = torch.zeros(len(e2g), self.hidden_channels).to(device)
        edge2graph = node2graph[data.edge_index[0]]
        row, col = data.edge_index
        for i, j, k in zip(row, col, e_emb):
            r = torch.nonzero(torch.eq(e_index[0], i)).view(-1)
            c = torch.nonzero(torch.eq(e_index[1], j)).view(-1)
            r_s = tuple(np.array(r.cpu()))
            c_s =tuple(np.array(c.cpu()))
            edge_index = set(r_s).intersection(set(c_s))
            for l in edge_index:
                e_tensors[l] = e_tensors[l] + k

        def collate_tokens(values, pad_idx, pad_to_multiple=1, left_pad=False, pad_to_length=None,):
            """Convert a list of 1d tensors into a padded 2d tensor."""
            size = max(v.size(0) for v in values)
            size = size if pad_to_length is None else max(size, pad_to_length)
            if pad_to_multiple != 1 and size % pad_to_multiple != 0:
                size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
            res = values[0].new(len(values), size, values[0].size(1)).fill_(pad_idx)
            def copy_tensor(src, dst):
                assert dst.numel() == src.numel()
                dst.copy_(src)
            for i, v in enumerate(values):
                copy_tensor(v, res[i][size - v.size(0):, :] if left_pad else res[i][: v.size(0), :])
            return res

        def collate_tokens_2d(values, pad_idx, left_pad=False, pad_to_length=None, pad_to_multiple=1,):
            """Convert a list of 1d tensors into a padded 2d tensor."""
            size = max(v.size(0) for v in values)
            size = size if pad_to_length is None else max(size, pad_to_length)
            if pad_to_multiple != 1 and size % pad_to_multiple != 0:
                size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
            res = values[0].new(len(values), size, size, values[0].size(2)).fill_(pad_idx)
            def copy_tensor(src, dst):
                assert dst.numel() == src.numel()
                dst.copy_(src)
            for i, v in enumerate(values):
                copy_tensor(v, res[i][size - len(v):, size - len(v):, :] if left_pad else res[i][:len(v), :len(v), :])
            return res

        j = 0
        k = 0
        n_values = []
        e_values = []
        for i in n_num:
            n_value = n_tensors[j: j+i]
            e_value = e_tensors[k: k+i*i].view(i, i, self.hidden_channels)
            j += i
            k = k + i*i
            n_values.append(n_value)
            e_values.append(e_value)
        n_tensors = collate_tokens(n_values, pad_idx=0, pad_to_multiple=8).to(device)
        e_tensors = collate_tokens_2d(e_values, pad_idx=0, pad_to_multiple=8).to(device)

        for layer in self.attention_layers:
            n_tensors, e_tensors = layer(n_tensors, e_tensors)

        return n_tensors

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_heads={self.num_heads}, "
            f"dropout={self.dropout}, "
            f"contrastive={self.contrastive}, "
            f"no_pos_encod={self.no_pos_encod}, "
        )