"""Sparse typed adaptation of author Diff Transformer V2 attention.

Paired query heads share their key/value head. Node/head-specific sigmoid
gates control subtraction. A common L1 coefficient normalization is an
explicit graph adaptation, used identically in the additive control.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import overlap_block_models as graph_lib
from improve.sharpness_graph_v1 import IdentityResidual, identity_task

KINDS = ('full', 'additive_attention')


def paired_weights(q, k, edges, edge_keys, lam, sign):
    """Q:N,H,2,D; K:N,H,D. Adjacent Q heads share one K/V group."""
    if sign not in (-1, 1):
        raise ValueError('Expected subtraction or addition')
    source, target = edges
    scores = (q[target] * (k[source] + edge_keys).unsqueeze(2)).sum(-1) / math.sqrt(k.shape[-1])
    attention = graph_lib.edge_softmax(scores.flatten(1), target, len(q)).reshape_as(scores)
    gate = lam[target]
    return (attention[:, :, 0] + sign * gate * attention[:, :, 1]) / (1 + gate)


def paired_aggregate(q, k, v, edges, edge_keys, lam, sign):
    weights = paired_weights(q, k, edges, edge_keys, lam, sign)
    output = v.new_zeros(v.shape)
    output.index_add_(0, edges[1], weights.unsqueeze(-1) * v[edges[0]])
    return output


class DifferentialAttention(nn.Module):
    def __init__(self, width, heads, edge_types, dropout, kind):
        super().__init__()
        self.heads, self.dim = heads, width // heads
        self.sign = -1 if kind == 'full' else 1
        self.q = nn.Linear(width, 2 * width, bias=False)
        self.k, self.v = nn.Linear(width, width, bias=False), nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width, heads, bias=False)
        self.edge = nn.Embedding(edge_types, width)
        nn.init.normal_(self.edge.weight, std=.02)
        self.output = nn.Linear(width, width)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2 * width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edges, relation):
        normalized = self.norm1(h)
        q = self.q(normalized).reshape(len(h), self.heads, 2, self.dim)
        k = self.k(normalized).reshape(len(h), self.heads, self.dim)
        v = self.v(normalized).reshape(len(h), self.heads, self.dim)
        edge_keys = self.edge(relation).reshape(-1, self.heads, self.dim)
        lam = self.gate(normalized).sigmoid()
        aggregate = paired_aggregate(q, k, v, edges, edge_keys, lam, self.sign)
        h = h + self.dropout(self.output(aggregate.flatten(1)))
        return h + self.dropout(self.ff(self.norm2(h)))


class DifferentialGraph(nn.Module):
    def __init__(self, dimensions, relations, p, kind):
        super().__init__()
        if kind not in KINDS or p['width'] % p['heads']:
            raise ValueError('Invalid differential configuration')
        width = p['width']
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dimensions])
        self.norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([DifferentialAttention(width, p['heads'], 2 * relations + 2,
            p['dropout'], kind) for _ in range(p['depth'])])
        self.readout = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(),
            nn.Dropout(p['dropout']), nn.Linear(width, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()

    def forward(self, xs, graph, pairs):
        edges, relation, counts = graph
        parts, start = [], 0
        for x, project in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            parts.append(project(torch.cat((x / rms, counts[start:start + len(x)]), -1)))
            start += len(x)
        h = F.gelu(self.norm(torch.cat(parts)))
        for layer in self.layers:
            h = layer(h, edges, relation)
        left, right = h[pairs[0]], h[pairs[1]]
        return self.readout(torch.cat((left, right, left * right, (left - right).abs()), -1)).squeeze(-1)


def parameters():
    return dict(width=64, heads=4, depth=2, dropout=.1, folds=5,
                attention_version='author_v2_shared_kv', coefficient_normalization='1+lambda',
                lambda_parameterization='node_head_sigmoid_projection',
                lnc_policy='identity_strong_reference')


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Unregistered differential arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        return IdentityResidual().to(device), [], [None] * p['folds'], folds
    n, relations = int(a['node_counts'].sum()), int(a['relation'].max()) + 1
    graphs = []
    for f in range(p['folds']):
        edges, labels = graph_lib.masked_physical(a, np.flatnonzero(a['folds'] == f))
        graphs.append(graph_lib.graph_inputs(edges, labels, np.empty((2, 0), dtype=np.int64),
                                            n, relations, device))
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = DifferentialGraph([x.shape[1] for x in xs], relations, p, kind).to(device)
    return model, xs, graphs, folds
