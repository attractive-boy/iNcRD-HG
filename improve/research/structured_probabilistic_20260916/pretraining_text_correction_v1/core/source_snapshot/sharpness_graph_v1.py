"""Shared ordinary graph backbone for a MiDrug-priority optimizer experiment.

The optimizer arms share exactly this architecture. The prospective LncDrug
deployment policy is an identity residual, preserving its frozen strong anchor.
No identity record is counted as a newly trained neural model.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import overlap_block_models as graph_lib

KINDS = ('full', 'sam', 'erm')


def target_type(a):
    offsets = np.r_[0, np.cumsum(a['node_counts'])[:-1]]
    rna = np.flatnonzero(offsets == a['pair_offset'][0])
    drug = np.flatnonzero(offsets == a['pair_offset'][1])
    if len(rna) != 1 or len(drug) != 1 or int(rna[0]) not in (0, 1) or int(drug[0]) != 2:
        raise ValueError('Expected frozen lncRNA/miRNA/drug dictionary order')
    return int(rna[0])


def identity_task(a, p):
    if p.get('lnc_policy') != 'identity_strong_reference':
        raise ValueError('Undeclared LncDrug deployment policy')
    return target_type(a) == 0


class IdentityResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()

    def forward(self, xs, graph, pairs):
        return torch.zeros(pairs.shape[1], dtype=torch.float32, device=pairs.device)


class OrdinaryGraph(nn.Module):
    def __init__(self, dimensions, relations, p):
        super().__init__()
        width = p['width']
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dimensions])
        self.norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([graph_lib.SparseTypedAttention(width, p['heads'],
            2 * relations + 2, p['dropout']) for _ in range(p['depth'])])
        self.readout = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(),
            nn.Dropout(p['dropout']), nn.Linear(width, 1))
        nn.init.zeros_(self.readout[-1].weight); nn.init.zeros_(self.readout[-1].bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()

    def forward(self, xs, graph, pairs):
        edges, relation, counts = graph
        parts, start = [], 0
        for x, project in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            parts.append(project(torch.cat((x / rms, counts[start:start + len(x)]), dim=-1)))
            start += len(x)
        h = F.gelu(self.norm(torch.cat(parts)))
        for layer in self.layers:
            h = layer(h, edges, relation)
        left, right = h[pairs[0]], h[pairs[1]]
        return self.readout(torch.cat((left, right, left * right, (left-right).abs()), dim=-1)).squeeze(-1)


def parameters():
    return dict(width=64, heads=4, depth=2, dropout=.1, folds=5,
                lnc_policy='identity_strong_reference')


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Unregistered optimizer arm')
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
    model = OrdinaryGraph([x.shape[1] for x in xs], relations, p).to(device)
    return model, xs, graphs, folds
