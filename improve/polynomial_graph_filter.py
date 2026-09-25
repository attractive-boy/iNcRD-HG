"""Query-masked, node-specific polynomial filtering inspired by PolyFormer.

KDD 2024, DOI 10.1145/3637528.3671849, arXiv:2407.14459v1.
This independent heterogeneous link-prediction adaptation forms Chebyshev
tokens from a learned typed field. It does not append features to a tabular
model and does not perform generative association-signal diffusion.
"""
from datetime import datetime, timezone
import math
import shutil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import overlap_block_experiment as common
from improve.overlap_block_models import masked_physical, graph_inputs

ROOT = common.base.ROOT
OUT = ROOT / 'results/polynomial_filter_core_20260915'
KINDS = ('full', 'type_shared', 'softmax', 'no_graph')


def normalized_operator(edges, n, device):
    arcs = np.unique(np.concatenate((edges, edges[::-1]), axis=1), axis=1)
    arcs = arcs[:, arcs[0] != arcs[1]]
    degrees = np.bincount(arcs[0], minlength=n).astype(np.float64)
    weights = (np.maximum(degrees[arcs[0]], 1.) * np.maximum(degrees[arcs[1]], 1.)) ** -.5
    return torch.sparse_coo_tensor(torch.as_tensor(arcs, device=device),
        torch.as_tensor(weights, dtype=torch.float32, device=device), (n, n), device=device).coalesce()


def context(a, fold, device):
    edges, relations = masked_physical(a, np.flatnonzero(a['folds'] == fold))
    n, nr = int(a['node_counts'].sum()), int(a['relation'].max() + 1)
    graph = graph_inputs(edges, relations, np.empty((2, 0), dtype=np.int64), n, nr, device)
    return normalized_operator(edges, n, device), graph[2]


def chebyshev_tokens(operator, field, order, no_graph=False):
    if order < 1:
        raise ValueError('A graph filter needs at least one nonconstant order')
    if no_graph:
        return torch.stack([field] + [torch.zeros_like(field) for _ in range(order)], 1)
    values = [field, torch.sparse.mm(operator, field)]
    for _ in range(2, order + 1):
        values.append(2 * torch.sparse.mm(operator, values[-1]) - values[-2])
    return torch.stack(values, 1)


def polynomial_attention(query, key, value, bias, kind, node_counts):
    """Attention over polynomial orders, never over the query-pair batch."""
    if kind not in KINDS:
        raise ValueError('Unknown polynomial filter arm')
    if kind == 'type_shared':
        def shared(v):
            return torch.cat([part.mean(0, keepdim=True).expand_as(part)
                              for part in v.split(node_counts, 0)], 0)
        query, key = shared(query), shared(key)
    logits = (query @ key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    scores = logits.softmax(-1) if kind == 'softmax' else logits.tanh()
    scores = scores * bias[None, :, None, :]
    return scores @ value, scores


class PolynomialBlock(nn.Module):
    def __init__(self, width, heads, order, dropout, kind, node_counts):
        super().__init__()
        self.heads, self.dimension, self.kind = heads, width // heads, kind
        self.node_counts = list(map(int, node_counts))
        self.orders = order + 1
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.order_mlp = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.ReLU(),
                                                    nn.Linear(width, width)) for _ in range(self.orders)])
        self.query, self.key = nn.Linear(width, width, bias=False), nn.Linear(width, width, bias=False)
        self.bias_scale = nn.Parameter(torch.ones(heads, self.orders))
        self.register_buffer('order_decay', 1 / torch.arange(1, self.orders + 1, dtype=torch.float32))
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        n = len(tokens)
        value = self.norm1(tokens)
        transformed = torch.stack([layer(value[:, j]) for j, layer in enumerate(self.order_mlp)], 1)
        def heads(v):
            return v.reshape(n, self.orders, self.heads, self.dimension).transpose(1, 2)
        mixed, _ = polynomial_attention(heads(self.query(transformed)), heads(self.key(transformed)),
            heads(value), self.bias_scale * self.order_decay[None, :], self.kind, self.node_counts)
        mixed = mixed.transpose(1, 2).reshape_as(tokens)
        tokens = tokens + self.dropout(mixed)
        return tokens + self.dropout(self.ff(self.norm2(tokens)))


class PolynomialGraphFilter(nn.Module):
    def __init__(self, dims, node_counts, relations, kind, width=64, depth=2, order=4, heads=4, dropout=.1):
        super().__init__()
        if kind not in KINDS or width % heads:
            raise ValueError('Invalid polynomial-filter configuration')
        self.kind, self.node_counts, self.order = kind, list(map(int, node_counts)), order
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dims])
        self.input_norm, self.output_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.layers = nn.ModuleList([PolynomialBlock(width, heads, order, dropout, kind, node_counts)
                                     for _ in range(depth)])
        self.rna_readout = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.drug_readout = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.rna_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))
        self.drug_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))
        self.scale = math.sqrt(width)

    def forward(self, xs, graph, pairs):
        operator, counts = graph
        pieces, offset = [], 0
        for x, project, size in zip(xs, self.project, self.node_counts):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            pieces.append(project(torch.cat((x / rms, counts[offset:offset + size]), 1)))
            offset += size
        field = F.gelu(self.input_norm(torch.cat(pieces)))
        tokens = chebyshev_tokens(operator, field, self.order, self.kind == 'no_graph')
        for layer in self.layers:
            tokens = layer(tokens)
        h = self.output_norm(tokens.sum(1))
        rna, drug = self.rna_readout(h), self.drug_readout(h)
        r, d = pairs
        score = (rna[r] * drug[d]).sum(-1) / self.scale
        return score + self.rna_degree(counts[r]).squeeze(-1) + self.drug_degree(counts[d]).squeeze(-1)


def parameters():
    return dict(kinds=list(KINDS), width=64, depth=2, order=4, heads=4, dropout=.1, folds=5,
                auxiliary_weight=0., primary_kind='full', primary_controls=['type_shared'])


def setup(a, p, kind, device):
    graphs = [context(a, fold, device) for fold in range(p['folds'])]
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == fold), device=device) for fold in range(p['folds'])]
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = PolynomialGraphFilter([x.shape[1] for x in xs], a['node_counts'], int(a['relation'].max() + 1),
        kind, p['width'], p['depth'], p['order'], p['heads'], p['dropout']).to(device)
    return model, xs, graphs, folds


def prepare():
    if OUT.exists():
        raise ValueError('Polynomial-filter specification already exists')
    parent = common.protocol(common.OUT)
    sources = ('polynomial_graph_filter.py', 'test_polynomial_graph_filter.py',
               'overlap_block_models.py', 'overlap_block_experiment.py', 'non_tabular_experiment.py')
    paper = ROOT / 'research/new_mechanisms_20260915/polyformer_paper.pdf'
    p = dict(**parameters(), created_at_utc=datetime.now(timezone.utc).isoformat(),
        tasks=parent['tasks'], selection=parent['selection'], specification_only=True,
        controls='Same graph, initialization, polynomial order, attention modules, decoder and training budget for all arms. type_shared averages Q/K within node type before forming attention, enforcing one filter per type. softmax is a sign-flexibility diagnostic; no_graph zeros nonconstant tokens and is a topology diagnostic. Both diagnostics are reported but do not define the primary comparison.',
        method='polynomial_filter_core', family='node_adaptive_spectral_filtering',
        literature=dict(title='PolyFormer: Scalable Node-wise Filters via Polynomial Graph Transformer',
                        venue='KDD 2024', doi='10.1145/3637528.3671849', arxiv='2407.14459v1'),
        adaptation='Independent heterogeneous link-prediction model, not the node-classification reproduction. Chebyshev recurrence on the symmetric normalized Boolean physical adjacency in [-1,1], typed learned input field, order 4, two polynomial-attention blocks, four heads, per-order decay 1/(k+1), endpoint dot decoder, output LayerNorm. Tokens are recomputed because typed input projections are trained. Uses tanh and sqrt(head dimension) scaling as in author code. Does not apply dropout to attention scores, only residual outputs.',
        boundary='Delete fitting query facts before counts, Boolean deduplication and the spectral operator. No heldout outcomes enter filtering. Type-shared attention uses the complete support graph, never the prediction-query batch. No numeric ID features.',
        input_policy='Same original node attributes and support graph as previous graph candidates; no extra tabular feature columns.',
        source_sha256={n: common.base.digest(ROOT / n) for n in sources},
        inputs={str(paper): common.base.digest(paper)},
        innovation_counting='One spectral-filter family, distinct from variational groups, geometric attention, state space, graph self-supervision, cavity messages and pair-conditioned neighborhood readout; not an extra generative diffusion result.')
    OUT.mkdir(parents=True)
    (OUT / 'source_snapshot').mkdir()
    for name in p['source_sha256']:
        shutil.copy2(ROOT / name, OUT / 'source_snapshot' / name)
    common.base.write_json(OUT / 'protocol.json', p)
