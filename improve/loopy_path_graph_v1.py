"""Learned simple-path messages over positive-support bipartite four-cycles.

Independent adaptation of Weisfeiler and Leman Go Loopy (NeurIPS 2024).
Ordinary typed physical attention is the shared backbone. The additional path
branch processes every ordered length-two path between distinct neighbors of a
center, excluding that center. Only the target association support induces loops.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from improve import overlap_block_models as graph_lib

KINDS = ('full', 'no_path_message')


def rooted_paths(pairs, rna_count, drug_count, offsets):
    """Every four-cycle once, then every root and both path orientations."""
    pairs = np.asarray(pairs, dtype=np.int64)
    b = np.zeros((rna_count, drug_count), dtype=bool)
    b[pairs[0], pairs[1]] = True
    cycles = []
    for left in range(drug_count):
        for right in range(left + 1, drug_count):
            shared = np.flatnonzero(b[:, left] & b[:, right])
            cycles.extend((r1 + offsets[0], left + offsets[1], r2 + offsets[0], right + offsets[1])
                          for r1, r2 in combinations(shared, 2))
    if not cycles:
        return np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.int64)
    cycles = np.asarray(cycles, dtype=np.int64)
    rooted = np.concatenate([np.roll(cycles, -shift, axis=1) for shift in range(4)])
    centers = np.tile(rooted[:, 0], 2)
    paths = np.concatenate((rooted[:, 1:], rooted[:, 1:][:, ::-1]))
    order = np.lexsort((paths[:, 2], paths[:, 1], paths[:, 0], centers))
    return centers[order], paths[order]


@dataclass
class LoopContext:
    edges: torch.Tensor
    relation: torch.Tensor
    counts: torch.Tensor
    centers: torch.Tensor
    paths: torch.Tensor
    path_degree: torch.Tensor
    support_positive_indices: np.ndarray
    omitted_indices: np.ndarray


def context(a, omitted, device):
    omitted = np.asarray(omitted, dtype=np.int64)
    support = np.ones(len(a['y_train']), dtype=bool); support[omitted] = False
    positive_ids = np.flatnonzero(support & (a['y_train'] == 1))
    pairs = a['pairs_train'][:, positive_ids]
    offsets = np.r_[0, np.cumsum(a['node_counts'])[:-1]]
    types = [int(np.flatnonzero(offsets == v)[0]) for v in a['pair_offset']]
    centers, paths = rooted_paths(pairs, *(int(a['node_counts'][i]) for i in types), a['pair_offset'])
    n, nr = int(a['node_counts'].sum()), int(a['relation'].max()) + 1
    physical, labels = graph_lib.masked_physical(a, omitted)
    query_codes = a['pairs_train'][0, omitted] * int(a['nd']) + a['pairs_train'][1, omitted]
    live_codes = a['target_codes'][(a['target_codes'] >= 0) & ~np.isin(a['target_codes'], query_codes)]
    np.testing.assert_array_equal(np.unique(live_codes),
        np.unique(pairs[0] * int(a['nd']) + pairs[1]))
    edges, relation, counts = graph_lib.graph_inputs(physical, labels,
        np.empty((2, 0), dtype=np.int64), n, nr, device)
    degree = np.bincount(centers, minlength=n)
    return LoopContext(edges, relation, counts, torch.as_tensor(centers, device=device),
        torch.as_tensor(paths, device=device), torch.as_tensor(degree, device=device),
        positive_ids, omitted.copy())


class PathGIN(nn.Module):
    def __init__(self, width, kind):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unregistered path mechanism')
        self.kind = kind
        self.distance = nn.Embedding(3, width)
        self.transform = nn.Linear(2 * width, width)
        self.epsilon = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, values):
        # In a support bipartite four-cycle the path has center distances 1,2,1.
        distances = torch.tensor([1, 2, 1], device=values.device)
        position = self.distance(distances)[None].expand(values.shape[0], -1, -1)
        transformed = self.transform(torch.cat((values, position), dim=-1))
        if self.kind == 'full':
            middle = transformed[:, 1]
            message = torch.stack((middle, transformed[:, 0] + transformed[:, 2], middle), dim=1)
        else:
            message = torch.zeros_like(transformed)
        # The transform is live in both arms. The control retains identical path
        # memberships and positional labels, removing only within-path messages.
        return self.mlp((1 + self.epsilon) * transformed + message).sum(1)


class LoopyLayer(nn.Module):
    def __init__(self, width, heads, relations, dropout, path_chunk, kind):
        super().__init__()
        self.physical = graph_lib.SparseTypedAttention(width, heads, 2 * relations + 2, dropout)
        self.norm = nn.LayerNorm(width)
        self.path = PathGIN(width, kind)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width),
                                    nn.GELU(), nn.Linear(2 * width, width))
        self.dropout = nn.Dropout(dropout)
        self.path_chunk = path_chunk

    def path_values(self, h, indices):
        return self.path(h[indices])

    def forward(self, h, graph):
        h = self.physical(h, graph.edges, graph.relation)
        normalized = self.norm(h)
        values = []
        for start in range(0, len(graph.paths), self.path_chunk):
            indices = graph.paths[start:start+self.path_chunk]
            if self.training and torch.is_grad_enabled():
                current = checkpoint(self.path_values, normalized, indices, use_reentrant=False,
                                     preserve_rng_state=False)
            else:
                current = self.path_values(normalized, indices)
            values.append(current)
        aggregate = torch.zeros_like(h)
        if values:
            aggregate.index_add_(0, graph.centers, torch.cat(values))
        aggregate = aggregate / graph.path_degree.clamp_min(1).sqrt()[:, None]
        active = graph.path_degree.ne(0)[:, None]
        return h + active * self.dropout(self.output(aggregate))


class LoopyPathGraph(nn.Module):
    def __init__(self, dimensions, relations, p, kind):
        super().__init__()
        self.kind = kind
        width = p['width']
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dimensions])
        self.input_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([LoopyLayer(width, p['heads'], relations, p['dropout'],
            p['path_chunk'], kind) for _ in range(p['depth'])])
        self.readout = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(),
            nn.Dropout(p['dropout']), nn.Linear(width, 1))
        nn.init.zeros_(self.readout[-1].weight); nn.init.zeros_(self.readout[-1].bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()

    def encode(self, xs, graph):
        projected, start = [], 0
        for x, layer in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            counts = graph.counts[start:start + len(x)]
            projected.append(layer(torch.cat((x / rms, counts), dim=-1)))
            start += len(x)
        h = F.gelu(self.input_norm(torch.cat(projected)))
        for layer in self.layers:
            h = layer(h, graph)
        return h

    def forward(self, xs, graph, pairs):
        h = self.encode(xs, graph)
        left, right = h[pairs[0]], h[pairs[1]]
        return self.readout(torch.cat((left, right, left * right, (left-right).abs()), dim=-1)).squeeze(-1)


def parameters():
    return dict(width=64, depth=2, heads=4, dropout=.1, folds=5, path_chunk=8192)


def setup(a, p, kind, device):
    graphs, folds = [], []
    for fold in range(p['folds']):
        omitted = np.flatnonzero(a['folds'] == fold)
        graphs.append(context(a, omitted, device))
        folds.append(torch.as_tensor(omitted, device=device))
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = LoopyPathGraph([x.shape[1] for x in xs], int(a['relation'].max()) + 1, p, kind).to(device)
    return model, xs, graphs, folds
