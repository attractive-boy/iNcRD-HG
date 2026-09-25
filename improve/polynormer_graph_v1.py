"""Typed, query-masked adaptation of Polynormer (ICLR 2024).

The author implementation's local product/skip update and sigmoid-kernel
global attention are retained. This is a link-prediction adaptation, not a
claim that a normalized, finite network satisfies the base-model theorem.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import overlap_block_models as graph_lib
from improve.sharpness_graph_v1 import IdentityResidual, identity_task

KINDS = ('full', 'additive_update')


def linear_attention(q, k, v):
    """Positive normalized attention in O(N H D^2), with N,H,D layout."""
    kv = torch.einsum('nhd,nhe->hde', k, v)
    numerator = torch.einsum('nhd,hde->nhe', q, kv)
    denominator = torch.einsum('nhd,hd->nh', q, k.sum(0))
    return numerator / denominator.clamp_min(1e-12).unsqueeze(-1)


class LocalAttention(nn.Module):
    def __init__(self, width, heads, edge_types):
        super().__init__()
        self.heads, self.dim = heads, width // heads
        self.value = nn.Linear(width, width, bias=False)
        self.att_source = nn.Parameter(torch.empty(heads, self.dim))
        self.att_target = nn.Parameter(torch.empty(heads, self.dim))
        nn.init.xavier_uniform_(self.att_source)
        nn.init.xavier_uniform_(self.att_target)
        self.relation_bias = nn.Embedding(edge_types, heads)
        nn.init.zeros_(self.relation_bias.weight)

    def forward(self, x, edges, relation):
        source, target = edges
        v = self.value(x).reshape(len(x), self.heads, self.dim)
        score = ((v[source] * self.att_source).sum(-1)
                 + (v[target] * self.att_target).sum(-1)
                 + self.relation_bias(relation))
        weights = graph_lib.edge_softmax(F.leaky_relu(score, .2), target, len(x))
        output = v.new_zeros(v.shape)
        output.index_add_(0, target, weights[..., None] * v[source])
        return output.flatten(1)


class LocalProduct(nn.Module):
    def __init__(self, width, heads, edge_types, dropout, kind):
        super().__init__()
        self.kind, self.dropout = kind, dropout
        self.attention = LocalAttention(width, heads, edge_types)
        self.self_value, self.factor = nn.Linear(width, width), nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.beta = nn.Parameter(torch.zeros(width))

    def forward(self, x, edges, relation):
        factor = F.relu(self.factor(x))
        value = F.dropout(F.relu(self.attention(x, edges, relation) + self.self_value(x)),
                          self.dropout, self.training)
        interaction = factor * value if self.kind == 'full' else factor + value
        beta = self.beta.sigmoid()
        return (1 - beta) * self.norm(interaction) + beta * value


class GlobalProducts(nn.Module):
    def __init__(self, width, heads, depth, dropout, kind):
        super().__init__()
        self.kind, self.heads, self.dim, self.dropout = kind, heads, width // heads, dropout
        self.keys = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.values = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.factors = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.beta = nn.Parameter(torch.zeros(depth, width))
        # The author's output projection is shared across global layers.
        self.output = nn.Linear(width, width)

    def forward(self, x):
        for i, (key, value, factor, norm) in enumerate(zip(self.keys, self.values, self.factors, self.norms)):
            qk = key(x).sigmoid().reshape(len(x), self.heads, self.dim)
            v = value(x).reshape(len(x), self.heads, self.dim)
            attended = norm(linear_attention(qk, qk, v).flatten(1))
            h = factor(x) + self.beta[i].sigmoid()
            interaction = attended * h if self.kind == 'full' else attended + h
            x = F.dropout(F.relu(self.output(interaction)), self.dropout, self.training)
        return x


class PolynomialGraph(nn.Module):
    def __init__(self, dimensions, relations, p, kind):
        super().__init__()
        if kind not in KINDS or p['width'] % p['heads']:
            raise ValueError('Invalid polynomial configuration')
        self.dropout = p['dropout']
        width = p['width']
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dimensions])
        self.input_norm, self.global_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.local = nn.ModuleList([LocalProduct(width, p['heads'], 2 * relations + 2,
            p['dropout'], kind) for _ in range(p['local_depth'])])
        self.global_layers = GlobalProducts(width, p['heads'], p['global_depth'], p['dropout'], kind)
        self.readout = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(),
            nn.Dropout(p['dropout']), nn.Linear(width, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()

    def encode(self, xs, graph):
        edges, relation, counts = graph
        parts, start = [], 0
        for x, project in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            parts.append(project(torch.cat((x / rms, counts[start:start + len(x)]), -1)))
            start += len(x)
        x = F.dropout(self.input_norm(torch.cat(parts)), self.dropout, self.training)
        local_sum = torch.zeros_like(x)
        for layer in self.local:
            x = layer(x, edges, relation)
            local_sum = local_sum + x
        return self.global_layers(self.global_norm(local_sum))

    def forward(self, xs, graph, pairs):
        h = self.encode(xs, graph)
        left, right = h[pairs[0]], h[pairs[1]]
        features = torch.cat((left, right, left * right, (left - right).abs()), -1)
        return self.readout(features).squeeze(-1)


def parameters():
    return dict(width=64, heads=4, local_depth=3, global_depth=2, dropout=.1,
                folds=5, beta_initial_logit=0., shared_query_key=True,
                lnc_policy='identity_strong_reference')


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Unregistered polynomial arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        return IdentityResidual().to(device), [], [None] * p['folds'], folds
    n, relations = int(a['node_counts'].sum()), int(a['relation'].max()) + 1
    graphs = []
    for f in range(p['folds']):
        edges, labels = graph_lib.masked_physical(a, np.flatnonzero(a['folds'] == f))
        arcs, types, counts = graph_lib.graph_inputs(edges, labels, np.empty((2, 0), dtype=np.int64),
                                                   n, relations, device)
        keep = arcs[0] != arcs[1]
        graphs.append((arcs[:, keep], types[keep], counts))
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = PolynomialGraph([x.shape[1] for x in xs], relations, p, kind).to(device)
    return model, xs, graphs, folds
