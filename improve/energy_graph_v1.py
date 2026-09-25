"""Bounded negative-edge energy descent inside a graph residual encoder.

Independent adaptation of YinYanGNN (TKDE 2025, arXiv:2310.09516v2).
Zeros are sampled association labels, never asserted biological antagonism.
The support's recorded zeros enter the forward energy; all query facts are
removed before either graph is constructed. No new zero labels are sampled.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import overlap_block_experiment as common
from improve import overlap_block_models as graph_lib


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/energy_core_v1_20260916'
KINDS = ('full', 'no_repulsion', 'attractive')


@dataclass
class NormalizedGraph:
    arcs: torch.Tensor
    weights: torch.Tensor
    degree: torch.Tensor
    active: torch.Tensor

    def laplacian(self, h):
        source, target = self.arcs
        aggregate = torch.zeros_like(h).index_add(
            0, target, h[source] * self.weights.to(h.dtype)[:, None])
        return self.active.to(h.dtype)[:, None] * h - aggregate


def normalized_graph(edges, n, device):
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError('Expected physical edge matrix')
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError('Physical edge is outside the node dictionary')
    canonical = np.sort(edges[:, edges[0] != edges[1]], axis=0)
    canonical = np.unique(canonical, axis=1)
    arcs = np.concatenate([canonical, canonical[::-1]], axis=1)
    degree = np.bincount(arcs[1], minlength=n).astype(np.float64)
    weights = 1. / np.sqrt(degree[arcs[0]] * degree[arcs[1]])
    return NormalizedGraph(torch.as_tensor(arcs, device=device),
        torch.as_tensor(weights, device=device), torch.as_tensor(degree, device=device),
        torch.as_tensor(degree > 0, device=device))


@dataclass
class EnergyContext:
    positive: NormalizedGraph
    negative: NormalizedGraph
    fidelity: torch.Tensor
    negative_training_indices: np.ndarray
    omitted_training_indices: np.ndarray


def context(a, omitted, device):
    omitted = np.asarray(omitted, dtype=np.int64)
    n = int(a['node_counts'].sum())
    positive_edges, _ = graph_lib.masked_physical(a, omitted)
    support = np.ones(len(a['y_train']), dtype=bool)
    support[omitted] = False
    negative_ids = np.flatnonzero(support & (a['y_train'] == 0))
    negative_edges = a['pairs_train'][:, negative_ids] + a['pair_offset'][:, None]
    positive = normalized_graph(positive_edges, n, device)
    negative = normalized_graph(negative_edges, n, device)
    fidelity = (positive.degree + negative.degree).clamp_min(1).reciprocal()
    return EnergyContext(positive, negative, fidelity, negative_ids, omitted.copy())


def energy_and_gradient(h, initial, graph, *, positive_weight=1., negative_weight=.5,
                        margin=1., temperature=.25, kind='full'):
    """A coherent scalar objective and its exact analytic gradient.

    The weighted fidelity norm uses D^(-1/2), matching the paper's D^(-1)
    gradient. Softplus is normalized by active negative nodes and channels to
    avoid graph-size-dependent saturation. The author code omits this bound;
    the present implementation explicitly retains the paper's bounded energy.
    """
    if kind not in KINDS or temperature <= 0:
        raise ValueError('Unknown arm or nonpositive energy temperature')
    positive = graph.positive.laplacian(h)
    negative = graph.negative.laplacian(h)
    fidelity = graph.fidelity.to(h.dtype)[:, None]
    difference = h - initial
    positive_trace = (h * positive).sum()
    negative_trace = (h * negative).sum()
    scale = graph.negative.active.sum().clamp_min(1).to(h.dtype) * h.shape[1]
    argument = (margin - negative_trace / scale) / temperature
    gate = torch.sigmoid(argument)
    sign = {'full': 1., 'no_repulsion': 0., 'attractive': -1.}[kind]
    energy = .5 * (fidelity * difference.square()).sum() + .5 * positive_weight * positive_trace
    energy = energy + sign * .5 * negative_weight * scale * temperature * F.softplus(argument)
    gradient = fidelity * difference + positive_weight * positive - sign * negative_weight * gate * negative
    return energy, gradient, gate


class EnergyGraph(nn.Module):
    def __init__(self, dimensions, parameters, kind='full'):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unknown energy mechanism arm')
        self.kind, self.parameters_config = kind, parameters
        width = parameters['width']
        self.project = nn.ModuleList([nn.Linear(d, width) for d in dimensions])
        self.input_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(parameters['dropout'])
        self.output = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(),
                                   nn.Dropout(parameters['dropout']), nn.Linear(width, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()

    def encode(self, xs, graph):
        values = []
        for x, project in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            values.append(project(x / rms))
        initial = self.dropout(torch.tanh(self.input_norm(torch.cat(values))))
        h, p = initial, self.parameters_config
        for _ in range(p['depth']):
            _, gradient, _ = energy_and_gradient(h, initial, graph, kind=self.kind,
                positive_weight=p['positive_weight'], negative_weight=p['negative_weight'],
                margin=p['margin'], temperature=p['energy_temperature'])
            h = h - p['step_size'] * gradient
        return h

    def forward(self, xs, graph, pairs):
        h = self.encode(xs, graph)
        r, d = h[pairs[0]], h[pairs[1]]
        return self.output(torch.cat([r, d, r * d, (r - d).abs()], dim=1)).squeeze(-1)


def setup(a, p, kind, device):
    graphs, folds = [], []
    for fold in range(p['folds']):
        omitted = np.flatnonzero(a['folds'] == fold)
        graphs.append(context(a, omitted, device))
        folds.append(torch.as_tensor(omitted, device=device))
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = EnergyGraph([x.shape[1] for x in xs], p, kind).to(device)
    return model, xs, graphs, folds


def parameters():
    return dict(width=64, depth=8, dropout=.1, folds=5, step_size=.2,
        positive_weight=1., negative_weight=.5, margin=1., energy_temperature=.25)


def prepare(out=OUT):
    out = Path(out)
    if out.exists():
        raise ValueError('Use a fresh energy core directory')
    names = ('energy_graph_v1.py', 'test_energy_graph_v1.py', 'overlap_block_models.py',
             'overlap_block_experiment.py', 'non_tabular_experiment.py')
    literature = ROOT / 'research/residual_next_20260916'
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='energy',
        implementation_module='energy_graph_v1', tasks=['MiDrug', 'LncDrug'],
        kinds=list(KINDS), primary_kind='full', primary_controls=['no_repulsion', 'attractive'],
        **parameters(), source_sha256={n: common.base.digest(ROOT / n) for n in names},
        inputs={str(literature / n): common.base.digest(literature / n) for n in
                ('energy_gnn_arxiv.pdf', 'ENERGY_AUTHOR_FILES_V1.json')},
        literature=dict(doi='10.1109/TKDE.2024.3481015', arxiv='2310.09516v2',
            title='Efficient Link Prediction via GNN Layers Induced by Negative Sampling',
            publication_year=2025, preprint_revision='2024-12-30'),
        selection='Two fixed learning rates, selected at seed 666 jointly across tasks; unchanged primary full.',
        mechanism='Differentiable unrolling of an energy with positive-edge attraction and bounded support-zero repulsion. The zero graph changes the forward encoder, not only the outer BCE loss.',
        controls='no_repulsion removes the zero-graph force while preserving fidelity scaling; attractive reverses that force using identical inputs, gate, parameter modules, and update count.',
        adaptations=['Typed raw-attribute projections and a binary residual decoder initialized to zero.',
            'Use only recorded support zero labels instead of resampling unobserved pairs.',
            'Undirected Boolean physical support; query facts deleted before both normalized Laplacians.',
            'Isolated-node normalized Laplacian is zero.',
            'Weighted fidelity is sum squared difference divided by combined degree, consistent with Eq. 12.',
            'Normalize the softplus argument by active negative nodes times width; finite temperature .25.',
            'Author yinyang.py uses an unbounded linear repulsion update; this version follows the bounded paper mechanism.',
            'No claim that the fixed finite step size guarantees energy descent at every trained state.'],
        boundary='The current neural query fold contributes no positive or zero facts. No query outcomes enter normalization, fidelity, or the energy gate.',
        novelty_scope='One forward energy mechanism family; no additional count for fixed anchor, scaling, or controls.',
        original_test_source_access=False, confirmed_innovations=0, smoke=False)
    out.mkdir(parents=True)
    (out / 'source_snapshot').mkdir()
    for name in names:
        shutil.copy2(ROOT / name, out / 'source_snapshot' / name)
    common.base.write_json(out / 'protocol.json', p)
    print('FROZEN_ENERGY_CORE', out, flush=True)
