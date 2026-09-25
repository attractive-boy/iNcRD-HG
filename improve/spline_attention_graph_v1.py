"""KAA-inspired spline scoring on fixed query-masked physical graphs.

Only the attention score mapping uses a KAN. A slightly larger MLP score
mapping is the mandatory control. The two models share every other parameter
initialization. Fixed cubic bases are not the paper's zero-order MRD theorem.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import overlap_block_models as graph_lib
from improve.sharpness_graph_v1 import IdentityResidual, identity_task

KINDS = ('full', 'mlp_score')


def cubic_basis(x, knots, degree=3):
    """Cox-de Boor basis, same half-open knot convention as scipy BSpline."""
    x = x.unsqueeze(-1)
    bases = ((x >= knots[:-1]) & (x < knots[1:])).to(x.dtype)
    for order in range(1, degree + 1):
        left = (x - knots[:-(order + 1)]) / (knots[order:-1] - knots[:-(order + 1)])
        right = (knots[order + 1:] - x) / (knots[order + 1:] - knots[1:-order])
        bases = left * bases[..., :-1] + right * bases[..., 1:]
    return bases


class SplineScore(nn.Module):
    def __init__(self, inputs, heads, grid_size=4, order=3):
        super().__init__()
        self.order = order
        self.register_buffer('knots', torch.arange(-order, grid_size + order + 1).float() / grid_size)
        self.base = nn.Parameter(torch.empty(heads, inputs))
        self.coefficients = nn.Parameter(torch.empty(heads, inputs, grid_size + order))
        self.bias = nn.Parameter(torch.zeros(heads))
        nn.init.uniform_(self.base, -1 / math.sqrt(inputs), 1 / math.sqrt(inputs))
        nn.init.normal_(self.coefficients, std=.02 / math.sqrt(inputs))

    def forward(self, aligned):
        basis = cubic_basis(aligned, self.knots, self.order)
        return (F.linear(F.silu(aligned), self.base, self.bias)
                + F.linear(basis.flatten(1), self.coefficients.flatten(1)))


class MLPScore(nn.Module):
    def __init__(self, inputs, heads, hidden=8):
        super().__init__()
        self.maps = nn.ModuleList([nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(),
                                                nn.Linear(hidden, 1)) for _ in range(heads)])

    def forward(self, aligned):
        return torch.cat([module(aligned) for module in self.maps], -1)


class SplineAttention(nn.Module):
    def __init__(self, width, heads, edge_types, dropout, kind, grid_size=4, order=3):
        super().__init__()
        self.heads, self.dim = heads, width // heads
        self.value = nn.Linear(width, width, bias=False)
        self.relation_bias = nn.Embedding(edge_types, heads)
        nn.init.zeros_(self.relation_bias.weight)
        self.output = nn.Linear(width, width)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2 * width, width))
        self.dropout = nn.Dropout(dropout)
        # Scorer-specific parameter counts must not perturb shared initialization.
        with torch.random.fork_rng(devices=[]):
            self.score = (SplineScore(width, heads, grid_size, order) if kind == 'full'
                          else MLPScore(width, heads))

    def forward(self, h, edges, relation):
        source, target = edges
        normalized = self.norm1(h)
        v = self.value(normalized).reshape(len(h), self.heads, self.dim)
        # Label-independent bounded alignment; both mappings see identical data.
        bounded = normalized.tanh()
        aligned = (bounded[target] - bounded[source]).abs() / 2
        score = self.score(aligned) + self.relation_bias(relation)
        weights = graph_lib.edge_softmax(score, target, len(h))
        aggregate = v.new_zeros(v.shape)
        aggregate.index_add_(0, target, weights[..., None] * v[source])
        h = h + self.dropout(self.output(aggregate.flatten(1)))
        return h + self.dropout(self.ff(self.norm2(h)))


class SplineGraph(nn.Module):
    def __init__(self, dimensions, relations, p, kind):
        super().__init__()
        if kind not in KINDS or p['width'] % p['heads']:
            raise ValueError('Invalid spline-attention configuration')
        width = p['width']
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dimensions])
        self.norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([SplineAttention(width, p['heads'], 2 * relations + 2,
            p['dropout'], kind, p['grid_size'], p['spline_order']) for _ in range(p['depth'])])
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
                grid_size=4, spline_order=3, grid_range=[0., 1.], mlp_hidden=8,
                grid_updates=False, lnc_policy='identity_strong_reference')


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Unregistered spline-attention arm')
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
    model = SplineGraph([x.shape[1] for x in xs], relations, p, kind).to(device)
    return model, xs, graphs, folds
