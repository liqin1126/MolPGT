import torch
import torch.nn as nn
import torch.nn.functional as F

class Basic_RT(nn.Module):
    def __init__(
            self,
            nb_heads: int,
            hidden_channels: int,
            dropout_rate: float,
            no_edge_update: bool
    ):
        super(Basic_RT, self).__init__()

        self.H = nb_heads
        self.HS = hidden_channels // nb_heads
        self.NS = hidden_channels
        self.ES = hidden_channels
        self.dropout_rate = dropout_rate
        self.no_edge_update = no_edge_update
        self.block = RTTransformerLayer(self.NS, self.H, self.HS,
                                   self.ES, self.dropout_rate, self.no_edge_update
                                   )

    def forward(self, node_tensors, edge_tensors):

        node_tensors, edge_tensors = self.block(node_tensors, edge_tensors)

        return node_tensors, edge_tensors


class RTTransformerLayer(nn.Module):
    def __init__(
            self,
            NS: int,
            H: int,
            HS: int,
            ES: int,
            dropout_rate: float,
            no_edge_update: bool,
    ):
        super(RTTransformerLayer, self).__init__()

        self.NS = NS
        self.H = H
        self.HS = HS
        self.ES = ES
        self.scale = 1.0 / torch.sqrt(torch.tensor(HS))
        self.dropout_rate = dropout_rate
        self.no_edge_update = no_edge_update
        self.Wnq = nn.Linear(self.NS, self.NS)
        self.Wnk = nn.Linear(self.NS, self.NS)
        self.Wnv = nn.Linear(self.NS, self.NS)
        self.Weq = nn.Linear(self.ES, self.ES)
        self.Wek = nn.Linear(self.ES, self.ES)
        self.Wev = nn.Linear(self.ES, self.ES)
        self.Wn1qe = nn.Linear(self.NS, self.NS)
        self.Wn1ke = nn.Linear(self.NS, self.NS)
        self.Wn1ve = nn.Linear(self.NS, self.NS)
        self.Wn2qe = nn.Linear(self.NS, self.NS)
        self.Wn2ke = nn.Linear(self.NS, self.NS)
        self.Wn2ve = nn.Linear(self.NS, self.NS)
        self.Weqe = nn.Linear(self.ES, self.ES)
        self.Weke = nn.Linear(self.ES, self.ES)
        self.Weve = nn.Linear(self.ES, self.ES)
        self.NL1 = nn.Linear(self.NS, self.NS)
        self.NLN1 = nn.LayerNorm(self.NS)
        self.NL2 = nn.Linear(self.NS, self.NS)
        self.EL1 = nn.Linear(self.ES, self.ES)
        self.ELN1 = nn.LayerNorm(self.ES)
        self.EL2 = nn.Linear(self.ES, self.ES)

        # Define the linear layers and normalization layers here

    def separate_node_heads(self, x):
        new_shape = x.size()[:-1] + (self.H, self.HS)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def separate_edge_heads(self, x):
        new_shape = x.size()[:-1] + (self.H, self.HS)
        x = x.view(new_shape)
        return x.permute(0, 3, 1, 2, 4)

    def concatenate_node_heads(self, x):
        x = x.permute(0, 2, 1, 3)
        new_shape = x.size()[:-2] + (self.NS,)
        return x.reshape(new_shape)

    def concatenate_edge_heads(self, x):
        x = x.permute(0, 2, 3, 1, 4)
        new_shape = x.size()[:-2] + (self.ES,)
        return x.reshape(new_shape)

    def forward(self, node_tensors, edge_tensors):

        device = node_tensors.device

        B = node_tensors.size(0)
        N = node_tensors.size(1)
        H = self.H
        HS = self.HS

        eQ = self.Weq(edge_tensors.to(device))
        eK = self.Wek(edge_tensors.to(device))
        eV = self.Wev(edge_tensors.to(device))

        nQ = self.Wnq(node_tensors)
        nK = self.Wnk(node_tensors)
        nV = self.Wnv(node_tensors)

        eQ = self.separate_edge_heads(eQ)
        eK = self.separate_edge_heads(eK)
        eV = self.separate_edge_heads(eV)

        nQ = self.separate_node_heads(nQ)
        nK = self.separate_node_heads(nK)
        nV = self.separate_node_heads(nV)

        Q = eQ + nQ.view(B, H, N, 1, HS)
        K = eK + nK.view(B, H, 1, N, HS)

        Q = Q.view(B, H, N, N, 1, HS)
        K = K.view(B, H, N, N, HS, 1)
        QK = torch.matmul(Q, K)
        QK = QK.view(B, H, N, N)

        QK = QK * self.scale.to(device)
        attn = F.softmax(QK, dim=-1)
        attn = attn.view(B, H, N, 1, N)

        V = eV + nV.view(B, H, 1, N, HS)
        new_nodes = torch.matmul(attn, V)
        new_nodes = new_nodes.view(B, H, N, HS)

        attw_node_tensors = self.concatenate_node_heads(new_nodes)

        res = self.NL2(F.relu(self.NL1(attw_node_tensors)))
        res = F.dropout(res, p=self.dropout_rate, training=self.training)
        node_tensors = self.NLN1(node_tensors + res)

        if not self.no_edge_update:
            n1Qe = self.Wn1qe(node_tensors)
            n1Ke = self.Wn1ke(node_tensors)
            n1Ve = self.Wn1ve(node_tensors)
            n2Qe = self.Wn2qe(node_tensors)
            n2Ke = self.Wn2ke(node_tensors)
            n2Ve = self.Wn2ve(node_tensors)

            eQe = self.Weqe(edge_tensors.to(device))
            eKe = self.Weke(edge_tensors.to(device))
            eVe = self.Weve(edge_tensors.to(device))

            eQe = self.separate_edge_heads(eQe)
            eKe = self.separate_edge_heads(eKe)
            eVe = self.separate_edge_heads(eVe)

            n1Qe = self.separate_node_heads(n1Qe)
            n1Ke = self.separate_node_heads(n1Ke)
            n1Ve = self.separate_node_heads(n1Ve)
            n2Qe = self.separate_node_heads(n2Qe)
            n2Ke = self.separate_node_heads(n2Ke)
            n2Ve = self.separate_node_heads(n2Ve)

            Qe = eQe + n1Qe.view(B, H, N, 1, HS) + n2Qe.view(B, H, 1, N, HS)
            Ke = eKe + n1Ke.view(B, H, N, 1, HS) + n2Ke.view(B, H, 1, N, HS)

            Qe = Qe.view(B, H, N, N, 1, HS)
            Ke = Ke.view(B, H, N, N, HS, 1)
            QKe = torch.matmul(Qe, Ke)
            QKe = QKe.view(B, H, N, N)

            QKe = QKe * self.scale.to(device)
            attn_e = nn.SiLU().to(device)(QKe)
            attn_e = attn_e.view(B, H, N, N, 1, 1)

            Ve = eVe + n1Ve.view(B, H, N, 1, HS) + n2Ve.view(B, H, 1, N, HS)
            Ve = Ve.view(B, H, N, N, 1, HS)
            new_edges = torch.matmul(attn_e, Ve)
            new_edges = new_edges.view(B, H, N, N, HS)

            attw_edge_tensors = self.concatenate_edge_heads(new_edges)

            res = self.EL2(F.relu(self.EL1(attw_edge_tensors)))
            res = F.dropout(res, p=self.dropout_rate, training=self.training)
            edge_tensors = self.ELN1(edge_tensors + res)
        else:
            edge_tensors = edge_tensors

        return node_tensors, edge_tensors

