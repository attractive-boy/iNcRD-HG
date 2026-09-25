"""Typed NCN pair decoding from learned common-neighbor representations.

Implements the observed-neighbor MPNN-then-SF mechanism of equation (9) in
Neural Common Neighbor with Completion for Link Prediction, ICLR 2024.
The completion algorithm in equations (10)-(11) is not implemented: the
task-specific scorer cannot assign probabilities to arbitrary relation types.
"""
from datetime import datetime, timezone
import math
import shutil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import overlap_block_experiment as common
from improve.overlap_block_models import masked_physical, graph_inputs, SparseTypedAttention

ROOT = common.base.ROOT
OUT = ROOT / 'results/neural_common_neighbor_core_20260915'
KINDS = ('full', 'count_only', 'node_only', 'union')


def context(a, fold, device):
    query = np.flatnonzero(a['folds'] == fold)
    edges, relations = masked_physical(a, query)
    n, nr = int(a['node_counts'].sum()), int(a['relation'].max() + 1)
    graph = graph_inputs(edges, relations, np.empty((2, 0), dtype=np.int64), n, nr, device)
    # This Boolean support is used only for set intersection. Learned message
    # passing retains relation types in graph. Parallel facts count once here.
    adjacency = torch.zeros((n, n), dtype=torch.bool, device=device)
    e = torch.as_tensor(edges, device=device)
    adjacency[e[0], e[1]] = True
    adjacency[e[1], e[0]] = True
    adjacency.fill_diagonal_(False)
    return (*graph, adjacency)


def neighbor_pool(adjacency, pairs, hidden, typed, kind):
    if kind not in KINDS:
        raise ValueError('Unknown neural common-neighbor arm')
    if kind == 'node_only':
        return hidden.new_zeros((pairs.shape[1], hidden.shape[1]))
    r, d = pairs
    mask = adjacency[r] | adjacency[d] if kind == 'union' else adjacency[r] & adjacency[d]
    weights = mask.to(hidden.dtype)
    count = weights.sum(1, keepdim=True).clamp_min(1)
    values = typed if kind == 'count_only' else hidden + typed
    # Square-root count normalization controls scale on the heterogeneous
    # graph; this is an explicit local adaptation of the paper's sum pooling.
    return (weights @ values) / count.sqrt()


class NeuralCommonNeighbor(nn.Module):
    def __init__(self, dims, node_counts, relations, kind, width=64, depth=3, dropout=.1):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unknown neural common-neighbor arm')
        self.node_counts, self.kind = list(node_counts), kind
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dims])
        self.input_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([SparseTypedAttention(width, 4, 2 * relations + 2, dropout) for _ in range(depth)])
        self.rna_readout = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.drug_readout = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.neighbor_readout = nn.Linear(width, width)
        self.type_value = nn.Embedding(len(node_counts), width)
        nn.init.normal_(self.type_value.weight, std=.1)
        self.pair_readout = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))
        self.rna_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))
        self.drug_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))
        node_types = torch.repeat_interleave(torch.arange(len(node_counts)), torch.as_tensor(node_counts))
        self.register_buffer('node_types', node_types)
        self.scale = math.sqrt(width)

    def encode(self, xs, graph):
        edges, relation, counts = graph[:3]
        pieces, offset = [], 0
        for x, project, size in zip(xs, self.project, self.node_counts):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            pieces.append(project(torch.cat((x / rms, counts[offset:offset + size]), 1)))
            offset += size
        h = F.gelu(self.input_norm(torch.cat(pieces)))
        for layer in self.layers:
            h = layer(h, edges, relation)
        return h

    def forward(self, xs, graph, pairs):
        h = self.encode(xs, graph)
        rna, drug = self.rna_readout(h), self.drug_readout(h)
        latent = self.neighbor_readout(h)
        typed = self.type_value(self.node_types)
        pooled = neighbor_pool(graph[3], pairs, latent, typed, self.kind)
        r, d = pairs
        interaction = self.pair_readout(torch.cat((rna[r] * drug[d] / self.scale, pooled), 1)).squeeze(-1)
        counts = graph[2]
        br, bd = self.rna_degree(counts).squeeze(-1), self.drug_degree(counts).squeeze(-1)
        return interaction + br[r] + bd[d]


def parameters():
    return dict(kinds=list(KINDS), width=64, depth=3, dropout=.1, folds=5,
                auxiliary_weight=0., primary_kind='full', primary_controls=['count_only', 'node_only'])


def setup(a, p, kind, device):
    graphs = [context(a, fold, device) for fold in range(p['folds'])]
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == fold), device=device) for fold in range(p['folds'])]
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = NeuralCommonNeighbor([x.shape[1] for x in xs], a['node_counts'], int(a['relation'].max() + 1),
        kind, p['width'], p['depth'], p['dropout']).to(device)
    return model, xs, graphs, folds


def prepare():
    if OUT.exists():
        raise ValueError('NCN core specification already exists')
    parent = common.protocol(common.OUT)
    sources = ('neural_common_neighbor.py', 'test_neural_common_neighbor.py',
               'overlap_block_models.py', 'overlap_block_experiment.py', 'non_tabular_experiment.py')
    paper = ROOT / 'research/new_mechanisms_20260915/ncn_paper.pdf'
    p = dict(**parameters(), created_at_utc=datetime.now(timezone.utc).isoformat(),
        tasks=parent['tasks'], selection=parent['selection'], specification_only=True,
        controls='All arms use identical support graphs, initialized modules, endpoint encoders and training budgets. count_only pools trainable type constants, exposing per-type common-neighbor counts but no learned neighbor representation. node_only zeros the pooling branch. union pools learned states from the neighborhood union as a diagnostic.',
        method='neural_common_neighbor_core',
        literature=dict(title='Neural Common Neighbor with Completion for Link Prediction',
                        venue='ICLR 2024', arxiv='2302.00890v4'),
        adaptation='Typed sparse-attention MPNN, separate endpoint projections, first-order common-neighbor pooling, trainable type constants and square-root count normalization. Fixed support likelihood supplies nuisance effects. This implements observed NCN only, not NCNC completion; no learned feature table is appended to the tabular reference.',
        family='pair_conditioned_neural_structural_decoding',
        boundary='Remove fitting query target facts before degree construction, inverse arcs and common-neighbor support. Validation outcomes do not enter any encoder or support. Node IDs are indices only; no numerical ID features.',
        source_sha256={n: common.base.digest(ROOT / n) for n in sources},
        inputs={str(paper): common.base.digest(paper)})
    OUT.mkdir(parents=True)
    (OUT / 'source_snapshot').mkdir()
    for name in sources:
        shutil.copy2(ROOT / name, OUT / 'source_snapshot' / name)
    common.base.write_json(OUT / 'protocol.json', p)
