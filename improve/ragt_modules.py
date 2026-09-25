"""Structural-augmented heterogeneous graph transformer backbone.

This is the new replacement backbone for the frozen ncRNA-drug protocol. It
replaces the released GeneralConv encoder with relation-aware heterogeneous
graph attention (HGT, Hu et al. 2020) and keeps the explicit, leak-free graph
pair statistics that the TabM arm used, so the relational operator does not
have to rediscover the degree signal that the released protocol mostly reads.

Two departures from a plain HGT are deliberate, because the project's own
history shows plain HGT/HAN/LightGCN lose here by 0.01-0.08 AUC:

* a per-relation in-degree encoding is added to every node input, so the
  magnitude information survives attention smoothing; and
* the decoder concatenates the cross-fitted structural pair features with the
  endpoint embeddings instead of relying on the embeddings alone.

This is a compact adaptation to the workspace's PyTorch 1.13 / PyG 2.6 stack,
not a reproduction of published benchmark numbers.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import HGTConv


class StructuralHGT(nn.Module):
    def __init__(self, node_dims, edge_types, rna_type, pair_dim, width=64,
                 depth=2, heads=4, dropout=.1, use_degree=True, fusion='concat',
                 gate_init=0.):
        super().__init__()
        if width % heads or depth < 1:
            raise ValueError('width must divide heads and depth must be positive')
        if fusion not in ('concat', 'add'):
            raise ValueError(fusion)
        self.node_types = tuple(node_dims)
        self.edge_types = tuple(edge_types)
        self.rna_type = rna_type
        self.width = width
        self.use_degree = bool(use_degree)
        self.pair_dim = int(pair_dim)
        self.fusion = fusion
        self.lin = nn.ModuleDict({nt: nn.LazyLinear(width) for nt in self.node_types})
        self.type_emb = nn.Embedding(len(self.node_types), width)
        if self.use_degree:
            self.degree = nn.Sequential(
                nn.Linear(len(self.edge_types), width), nn.LeakyReLU(.1),
                nn.Linear(width, width))
        metadata = (list(self.node_types), list(self.edge_types))
        self.convs = nn.ModuleList([
            HGTConv(width, width, metadata, heads=heads) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.dropout = nn.Dropout(dropout)
        if self.fusion == 'add':
            self.feature_mlp = nn.Sequential(
                nn.Linear(pair_dim, width), nn.LeakyReLU(.1), nn.Dropout(dropout),
                nn.Linear(width, width), nn.LeakyReLU(.1), nn.Linear(width, 1))
            self.graph_mlp = nn.Sequential(
                nn.Linear(4*width, 2*width), nn.LeakyReLU(.1), nn.Dropout(dropout),
                nn.Linear(2*width, width), nn.LeakyReLU(.1), nn.Linear(width, 1))
            self.graph_gate = nn.Parameter(torch.zeros(1) + float(gate_init))
        else:
            in_dim = 4*width + self.pair_dim
            self.decoder = nn.Sequential(
                nn.Linear(in_dim, 2*width), nn.LeakyReLU(.1), nn.Dropout(dropout),
                nn.Linear(2*width, width), nn.LeakyReLU(.1), nn.Linear(width, 1))

    def _degrees(self, x_dict, edge_index_dict):
        """Per-relation in-degree of every node, zero where it is not a target."""
        rows = {}
        for nt in self.node_types:
            rows[nt] = x_dict[nt].new_zeros((x_dict[nt].size(0), len(self.edge_types)))
        for i, key in enumerate(self.edge_types):
            edge = edge_index_dict.get(key)
            if edge is None or edge.numel() == 0:
                continue
            dst = key[2]
            rows[dst][:, i] = torch.bincount(
                edge[1], minlength=x_dict[dst].size(0)).to(rows[dst].dtype)
        return {nt: torch.log1p(value) for nt, value in rows.items()}

    def encode(self, x_dict, edge_index_dict):
        if tuple(x_dict) != self.node_types:
            raise ValueError('Node-type order differs from the trained model')
        degree = self._degrees(x_dict, edge_index_dict) if self.use_degree else None
        h = {}
        for i, nt in enumerate(self.node_types):
            z = self.lin[nt](x_dict[nt]) + self.type_emb.weight[i]
            if degree is not None:
                z = z + self.degree(degree[nt])
            h[nt] = z
        for conv, norm in zip(self.convs, self.norms):
            update = conv(h, edge_index_dict)
            for nt in self.node_types:
                value = update.get(nt)
                if value is None:
                    continue
                h[nt] = h[nt] + self.dropout(F.leaky_relu(norm(value), .1))
        return h

    def decode(self, z, pairs, features):
        r, d = pairs
        zr, zd = z[self.rna_type][r], z['drug'][d]
        if self.fusion == 'add':
            base = self.feature_mlp(features)
            graph = self.graph_mlp(torch.cat([zr, zd, (zr-zd).abs(), zr*zd], dim=-1))
            return (base + self.graph_gate.sigmoid() * graph).view(-1)
        joint = torch.cat([zr, zd, (zr-zd).abs(), zr*zd, features], dim=-1)
        return self.decoder(joint).view(-1)

    def forward(self, x_dict, edge_index_dict, pairs, features):
        return self.decode(self.encode(x_dict, edge_index_dict), pairs, features)
