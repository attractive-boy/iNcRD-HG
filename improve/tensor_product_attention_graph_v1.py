"""Contextual tensor-product Q/K/V in support-only typed graph attention.

The CP factor layout and initialization follow tensorgi/TPA at commit
c276c80d5ad807881dedb4707d8d3c20b4e97ec6. RoPE is omitted because node
identifiers are not sequence positions. We materialize Q/K/V and compute
all control branches, so no autoregressive cache or speedup is claimed.

Adapted portions are covered by the MIT License:
Copyright (c) 2025 tensorgi
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import nested_anchor_inputs_v2 as nested
from improve import overlap_block_models as graph_lib
from improve.sharpness_graph_v1 import IdentityResidual, identity_task

KINDS = ('full', 'noncontextual_a', 'dense_qkv')
TRAIN_KEYS = nested.TRAIN_KEYS | {'anchor_oof'}


def parameters():
    return dict(width=64, heads=4, depth=2, dropout=.1, folds=5, q_rank=6, kv_rank=2,
        factor_initialization='pinned CPLinear three-dimensional Xavier views',
        static_a_initialization='initial contextual A evaluated at an all-ones vector',
        dense_initialization='exact effective linear maps of initial noncontextual-A arm',
        rank_normalization='1/rank for each Q/K/V product',
        positional_encoding='none; node IDs are not sequence positions',
        lnc_policy='identity_strong_reference')


def empty_diagnostics():
    return dict(calls=0, attention_layers=0, node_rows=0, contextual_product_terms=0,
        static_product_terms=0, dense_projection_outputs=0, scored_edge_heads=0,
        value_message_entries=0, maximum_abs_attention_score=0., maximum_abs_state=0.)


class TensorProductQKV(nn.Module):
    def __init__(self, in_features, heads, head_dim, q_rank=6, kv_rank=2, kind='full'):
        super().__init__()
        if kind not in KINDS or min(in_features, heads, head_dim, q_rank, kv_rank) < 1:
            raise ValueError('Invalid tensor-product arm or dimensions')
        self.in_features, self.heads, self.head_dim = in_features, heads, head_dim
        self.q_rank, self.kv_rank, self.kind = q_rank, kv_rank, kind
        self.ranks = dict(q=q_rank, k=kv_rank, v=kv_rank)
        # Preserve the pinned author's six Linear construction and reset order.
        for letter, rank in self.ranks.items():
            setattr(self, 'W_A_'+letter, nn.Linear(in_features, heads*rank, bias=False))
        for letter, rank in self.ranks.items():
            setattr(self, 'W_B_'+letter, nn.Linear(in_features, rank*head_dim, bias=False))
        for letter, rank in self.ranks.items():
            value = getattr(self, 'W_A_'+letter).weight
            nn.init.xavier_uniform_(value.view(in_features, heads, rank))
        for letter, rank in self.ranks.items():
            value = getattr(self, 'W_B_'+letter).weight
            nn.init.xavier_uniform_(value.view(in_features, rank, head_dim))
        with torch.no_grad():
            for letter, rank in self.ranks.items():
                wa = getattr(self, 'W_A_'+letter).weight
                wb = getattr(self, 'W_B_'+letter).weight
                constant = wa.sum(-1).reshape(heads, rank)
                dense = torch.einsum('hr,rdi->hdi', constant,
                    wb.reshape(rank, head_dim, in_features)).reshape(heads*head_dim, in_features)/rank
                setattr(self, 'static_A_'+letter, nn.Parameter(constant.clone()))
                setattr(self, 'dense_W_'+letter, nn.Parameter(dense.clone()))
        for name, value in self.named_parameters():
            active = (kind == 'full' and name.startswith(('W_A_', 'W_B_'))
                or kind == 'noncontextual_a' and name.startswith(('static_A_', 'W_B_'))
                or kind == 'dense_qkv' and name.startswith('dense_W_'))
            value.requires_grad_(active)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError('Expected node-by-feature tensor')
        output = []
        for letter, rank in self.ranks.items():
            a = getattr(self, 'W_A_'+letter)(x).reshape(-1, self.heads, rank)
            b = getattr(self, 'W_B_'+letter)(x).reshape(-1, rank, self.head_dim)
            contextual = torch.bmm(a, b)/rank
            static = torch.matmul(getattr(self, 'static_A_'+letter), b)/rank
            dense = F.linear(x, getattr(self, 'dense_W_'+letter)).reshape(-1, self.heads, self.head_dim)
            if self.kind == 'full':
                value = contextual + 0.*(static+dense)
            elif self.kind == 'noncontextual_a':
                value = static + 0.*(contextual+dense)
            else:
                value = dense + 0.*(contextual+static)
            if not torch.isfinite(value).all():
                raise FloatingPointError('Nonfinite tensor-product projection')
            output.append(value)
        return tuple(output)


class TensorProductAttention(nn.Module):
    def __init__(self, width, heads, edge_types, dropout, q_rank, kv_rank, kind):
        super().__init__()
        if width % heads:
            raise ValueError('Width must divide into heads')
        self.heads, self.dim = heads, width//heads
        self.qkv = TensorProductQKV(width, heads, self.dim, q_rank, kv_rank, kind)
        self.edge = nn.Embedding(edge_types, width)
        nn.init.normal_(self.edge.weight, std=.02)
        self.output = nn.Linear(width, width)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2*width, width))
        self.dropout = nn.Dropout(dropout)

    def aggregate(self, q, k, v, edges, relation):
        src, dst = edges
        edge = self.edge(relation).reshape(-1, self.heads, self.dim)
        score = (q[dst]*(k[src]+edge)).sum(-1)/math.sqrt(self.dim)
        if not torch.isfinite(score).all():
            raise FloatingPointError('Nonfinite tensor-product attention score')
        attention = graph_lib.edge_softmax(score, dst, len(q))
        result = q.new_zeros((len(q), self.heads, self.dim))
        result.index_add_(0, dst, attention[..., None]*v[src])
        return result, score

    def forward(self, h, edges, relation, diagnostics):
        q, k, v = self.qkv(self.norm1(h))
        aggregate, score = self.aggregate(q, k, v, edges, relation)
        h = h + self.dropout(self.output(aggregate.flatten(1)))
        h = h + self.dropout(self.ff(self.norm2(h)))
        if not torch.isfinite(h).all():
            raise FloatingPointError('Nonfinite tensor-product graph state')
        d = diagnostics
        nodes, edge_count = len(h), edges.shape[1]
        terms = nodes*self.heads*self.dim*(self.qkv.q_rank+2*self.qkv.kv_rank)
        d['attention_layers'] += 1
        d['node_rows'] += nodes
        d['contextual_product_terms'] += terms
        d['static_product_terms'] += terms
        d['dense_projection_outputs'] += nodes*3*self.heads*self.dim
        d['scored_edge_heads'] += edge_count*self.heads
        d['value_message_entries'] += edge_count*self.heads*self.dim
        d['maximum_abs_attention_score'] = max(d['maximum_abs_attention_score'], float(score.detach().abs().max()))
        d['maximum_abs_state'] = max(d['maximum_abs_state'], float(h.detach().abs().max()))
        return h


class TensorProductGraph(nn.Module):
    def __init__(self, dimensions, relations, p, kind):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unregistered tensor-product arm')
        width = p['width']
        self.project = nn.ModuleList([nn.Linear(d+relations, width) for d in dimensions])
        self.norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([TensorProductAttention(width, p['heads'],
            2*relations+2, p['dropout'], p['q_rank'], p['kv_rank'], kind) for _ in range(p['depth'])])
        self.readout = nn.Sequential(nn.Linear(4*width, width), nn.GELU(),
            nn.Dropout(p['dropout']), nn.Linear(width, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()
        self.tpa_diagnostics = empty_diagnostics()

    def forward(self, xs, graph, pairs):
        edges, relation, counts = graph
        parts, start = [], 0
        for x, projection in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            parts.append(projection(torch.cat((x/rms, counts[start:start+len(x)]), dim=-1)))
            start += len(x)
        h = F.gelu(self.norm(torch.cat(parts)))
        self.tpa_diagnostics['calls'] += 1
        for layer in self.layers:
            h = layer(h, edges, relation, self.tpa_diagnostics)
        left, right = h[pairs[0]], h[pairs[1]]
        return self.readout(torch.cat((left, right, left*right, (left-right).abs()), dim=-1)).squeeze(-1)


def setup(a, p, kind, device):
    if set(a)-TRAIN_KEYS:
        raise ValueError('Tensor-product setup accepts training arrays only')
    if kind not in KINDS:
        raise ValueError('Unregistered tensor-product arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        return IdentityResidual().to(device), [], [None]*p['folds'], folds
    n, relations = int(a['node_counts'].sum()), int(a['relation'].max())+1
    graphs = []
    for f in range(p['folds']):
        edges, labels = graph_lib.masked_physical(a, np.flatnonzero(a['folds'] == f))
        graphs.append(graph_lib.graph_inputs(edges, labels,
            np.empty((2, 0), dtype=np.int64), n, relations, device))
    xs = [torch.as_tensor(a['x_'+str(i)], device=device) for i in range(len(a['node_counts']))]
    model = TensorProductGraph([x.shape[1] for x in xs], relations, p, kind).to(device)
    return model, xs, graphs, folds
