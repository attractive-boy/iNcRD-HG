"""Task-trained continuous latent adjacency inspired by DGM (TPAMI 2023).

This is an independent heterogeneous link-prediction adaptation of cDGM.
Latent edges connect nodes of the same type and are not biological facts.
All physical query facts are removed before the fields defining adjacency.
"""
from datetime import datetime, timezone
import math
import shutil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import overlap_block_experiment as common
from improve.overlap_block_models import SparseTypedAttention, graph_inputs, masked_physical

ROOT = common.base.ROOT
OUT = ROOT / 'results/learned_structure_core_20260915'
KINDS = ('full', 'detached', 'static', 'no_rewire')


def distance_adjacency(field, inverse_scale, threshold):
    """Continuous cDGM affinities: nearer nodes have larger positive weight.

    Unit-normalized fields bound squared distances to [0,4]. The diagonal
    is removed and each nonempty row sums to one. A singleton has no edges.
    """
    z = F.normalize(field, dim=-1, eps=1e-8)
    sq = z.square().sum(-1)
    distance = (sq[:, None] + sq[None, :] - 2 * (z @ z.T)).clamp_min(0.)
    weights = torch.sigmoid(inverse_scale * (threshold - distance))
    weights = weights * (1 - torch.eye(len(z), device=z.device, dtype=z.dtype))
    return weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(z.dtype).tiny)


def context(a, fold, device, static_adjacency):
    edges, relations = masked_physical(a, np.flatnonzero(a['folds'] == fold))
    n, nr = int(a['node_counts'].sum()), int(a['relation'].max() + 1)
    physical = graph_inputs(edges, relations, np.empty((2, 0), dtype=np.int64), n, nr, device)
    return physical, static_adjacency


class StructureLayer(nn.Module):
    def __init__(self, width, heads, edge_types, metric_width, node_counts, dropout, kind):
        super().__init__()
        self.kind, self.node_counts = kind, list(map(int, node_counts))
        self.physical = SparseTypedAttention(width, heads, edge_types, dropout)
        self.norm = nn.LayerNorm(width)
        self.metric = nn.Sequential(nn.Linear(width, width // 2), nn.GELU(),
                                    nn.Linear(width // 2, metric_width))
        self.value = nn.Linear(width + metric_width, width)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.log_inverse_scale = nn.Parameter(torch.full((len(node_counts),), math.log(math.expm1(3.))))
        self.raw_threshold = nn.Parameter(torch.full((len(node_counts),), math.log(1 / 3)))

    def latent(self, hidden, static_adjacency):
        normalized = self.norm(hidden)
        metric = self.metric(normalized)
        # This value path is present in every arm, including no_rewire. Thus
        # detached adjacency leaves a trainable metric network of equal size.
        values = self.value(torch.cat((normalized, metric), -1))
        pieces = []
        offset = 0
        for i, size in enumerate(self.node_counts):
            features = metric[offset:offset + size]
            value = values[offset:offset + size]
            if self.kind == 'no_rewire':
                message = value
            else:
                if self.kind == 'static':
                    adjacency = static_adjacency[i]
                else:
                    adjacency = distance_adjacency(features,
                        F.softplus(self.log_inverse_scale[i]) + 1e-4,
                        4 * self.raw_threshold[i].sigmoid())
                    if self.kind == 'detached':
                        adjacency = adjacency.detach()
                message = adjacency @ value
            pieces.append(message)
            offset += size
        return hidden + self.dropout(self.output(torch.cat(pieces)))

    def forward(self, hidden, graph):
        (edges, relations, _), static_adjacency = graph
        hidden = self.physical(hidden, edges, relations)
        return self.latent(hidden, static_adjacency)


class LearnedGraphStructure(nn.Module):
    def __init__(self, dims, node_counts, relations, kind, width=64, depth=2,
                 heads=4, metric_width=16, dropout=.1):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unknown structure-learning arm')
        self.node_counts = list(map(int, node_counts))
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dims])
        self.input_norm, self.output_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.layers = nn.ModuleList([StructureLayer(width, heads, 2 * relations + 2,
            metric_width, node_counts, dropout, kind) for _ in range(depth)])
        self.rna_readout = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.drug_readout = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.rna_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))
        self.drug_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))
        self.scale = math.sqrt(width)

    def forward(self, xs, graph, pairs):
        counts = graph[0][2]
        pieces, offset = [], 0
        for x, project, size in zip(xs, self.project, self.node_counts):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            pieces.append(project(torch.cat((x / rms, counts[offset:offset + size]), 1)))
            offset += size
        hidden = F.gelu(self.input_norm(torch.cat(pieces)))
        for layer in self.layers:
            hidden = layer(hidden, graph)
        hidden = self.output_norm(hidden)
        r, d = pairs
        score = (self.rna_readout(hidden)[r] * self.drug_readout(hidden)[d]).sum(-1) / self.scale
        return score + self.rna_degree(counts[r]).squeeze(-1) + self.drug_degree(counts[d]).squeeze(-1)


def parameters():
    return dict(kinds=list(KINDS), width=64, depth=2, heads=4, metric_width=16,
                dropout=.1, folds=5, auxiliary_weight=0., primary_kind='full',
                primary_controls=['detached', 'static', 'no_rewire'])


def setup(a, p, kind, device):
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    # Static control depends only on original attributes, never query labels
    # or node/query ordering. Shared immutable tensors avoid five dense copies.
    with torch.no_grad():
        static = tuple(distance_adjacency(x, 3.0001, 1.) for x in xs)
    graphs = [context(a, fold, device, static) for fold in range(p['folds'])]
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == fold), device=device) for fold in range(p['folds'])]
    model = LearnedGraphStructure([x.shape[1] for x in xs], a['node_counts'], int(a['relation'].max() + 1),
        kind, p['width'], p['depth'], p['heads'], p['metric_width'], p['dropout']).to(device)
    return model, xs, graphs, folds


def prepare():
    if OUT.exists():
        raise ValueError('Use a new structure-learning specification directory')
    parent = common.protocol(common.OUT)
    sources = ('learned_graph_structure.py', 'test_learned_graph_structure.py',
               'overlap_block_models.py', 'overlap_block_experiment.py', 'non_tabular_experiment.py')
    paper = ROOT / 'research/new_mechanisms_20260915/dgm_paper.pdf'
    author = ROOT / 'research/new_mechanisms_20260915/dgm_author_layers.py'
    p = dict(**parameters(), created_at_utc=datetime.now(timezone.utc).isoformat(),
        tasks=parent['tasks'], selection=parent['selection'], specification_only=True,
        method='learned_structure_core', family='differentiable_graph_structure_learning',
        literature=dict(title='Differentiable Graph Module (DGM) for Graph Convolutional Networks',
            doi='10.1109/TPAMI.2022.3170249', venue='TPAMI 2023', arxiv='2002.04999v4'),
        controls='All arms have identical modules, initialization and a common trainable metric-to-value path. detached blocks only adjacency gradients, static uses the same continuous affinity formula on original attributes, no_rewire uses the per-node value with no latent mixing. The full model must beat all three controls on mean MiDrug AUC/AP, as well as the offset-only null.',
        adaptation='Two alternating physical sparse-attention and cDGM layers, within-type dense latent adjacency, metric width 16, unit-normalized learned graph features, positive inverse scale and bounded threshold, zero diagonal and row normalization. Metric features also enter value transforms in every arm for a capacity-matched no-rewire comparison. Physical edges remain a separate observed-relation channel. This is a link-prediction adaptation, not an official reproduction or sparse dDGM/Gumbel sampler.',
        affinity_sign='Use sigmoid(t*(T-distance_squared)), matching author DGM_c code. The PDF prints the opposite sign in its continuous formula. Temperature is constrained positive locally so farther nodes cannot receive larger affinity. Threshold is 4*sigmoid(raw), locally bounded because metric features are unit-normalized.',
        boundary='Remove queried physical facts before relation counts and hidden fields. Generate latent adjacency only within a node type, so it cannot reinsert an RNA-drug target fact. Recompute from the complete support graph for inference, never query-batch statistics. Latent edges are model constructs, not verified biology.',
        input_policy='Original attributes and masked physical graph only; no extra tabular columns, external data, pretrained embeddings or frozen feature-kernel predictor.',
        source_sha256={n: common.base.digest(ROOT / n) for n in sources},
        inputs={str(x): common.base.digest(x) for x in (paper, author)},
        innovation_counting='One learned-topology family. The fixed offset, similarity control and logit composition do not count as innovations.')
    OUT.mkdir(parents=True)
    (OUT / 'source_snapshot').mkdir()
    for name in p['source_sha256']:
        shutil.copy2(ROOT / name, OUT / 'source_snapshot' / name)
    common.base.write_json(OUT / 'protocol.json', p)
