"""Conditional determinantal selection from complete query-neighborhood unions.

Uses CDPruner's quality-times-Gram subset objective, adapted to graph evidence.
The fixed-cardinality greedy rule is a MAP heuristic, not exact DPP sampling or
a claimed approximation guarantee. No gradient is invented for hard selection.
"""
from collections import OrderedDict

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F

from improve import nested_anchor_inputs_v2 as nested
from improve import overlap_block_models as graph_lib
from improve import sharpness_graph_v1 as ordinary

KINDS = ('full', 'no_diversity', 'no_condition')
TRAIN_KEYS = nested.TRAIN_KEYS | {'anchor_oof'}
identity_task = ordinary.identity_task


def parameters():
    return dict(**ordinary.parameters(), retained_neighbors=16, quality_floor=1e-4,
        gram_diagonal_ridge=1e-5, selector_dtype='float64',
        neighborhood='all one-hop union neighbors, excluding both query endpoints',
        candidate_node_cap=None, relevance='positive cosine to the sum of the two endpoint embeddings',
        constant_relevance='uniform quality', tie_rule='lexicographic hidden feature and membership')


def normalize_rows(value):
    if not torch.isfinite(value).all():
        raise FloatingPointError('Nonfinite feature before cosine normalization')
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    return torch.where(norm > 0, value / norm.clamp_min(torch.finfo(value.dtype).tiny), torch.zeros_like(value))


@torch.no_grad()
def conditional_quality(features, query, groups, number, floor):
    if not 0. <= floor < 1.:
        raise ValueError('Invalid relevance floor')
    relevance = (features * query[groups]).sum(-1)
    lo = relevance.new_full((number,), torch.inf).scatter_reduce_(0, groups, relevance, reduce='amin')
    hi = relevance.new_full((number,), -torch.inf).scatter_reduce_(0, groups, relevance, reduce='amax')
    span = (hi - lo)[groups]
    scaled = torch.where(span > 1e-12, (relevance - lo[groups]) / span.clamp_min(1e-12),
                         torch.ones_like(relevance))
    return floor + (1. - floor) * scaled.clamp(0., 1.)


@torch.no_grad()
def greedy_subset(features, quality, groups, priority, number, budget, ridge, *, diagonal=False):
    """Ragged, matrix-free Cholesky greedy selection for diag(q)(ZZ'+ridge I)diag(q).

    Inputs are unit feature rows (or exact zero rows). All potential neighbors
    enter the objective. Only one kernel row per selected item is formed.
    """
    m = len(features)
    if (features.ndim != 2 or quality.shape != (m,) or groups.shape != (m,) or priority.shape != (m,)
            or features.dtype != torch.float64 or quality.dtype != torch.float64
            or groups.dtype != torch.int64 or priority.dtype != torch.int64
            or budget < 1 or number < 1 or ridge < 0):
        raise ValueError('Invalid determinantal selector inputs')
    if (not torch.isfinite(features).all() or not torch.isfinite(quality).all()
            or not torch.all(quality > 0) or (m and (int(groups.min()) < 0 or int(groups.max()) >= number))):
        raise ValueError('Nonfinite or invalid determinant features/groups')
    counts = torch.bincount(groups, minlength=number)
    selected = torch.full((number, budget), -1, dtype=torch.long, device=features.device)
    logdet = features.new_zeros(number)
    pivots = features.new_full((number, budget), torch.nan)
    if m == 0:
        return dict(indices=selected, counts=counts, logdet=logdet, pivots=pivots)
    norm2 = features.square().sum(1)
    if not torch.all((norm2 < 1e-20) | ((norm2 - 1.).abs() < 1e-10)):
        raise ValueError('Selector features must be normalized, or zero')
    # Cosine diagonals are mathematically 1. Fix their rounding so that unit-row
    # norm roundoff does not become an unintended tie breaker for no_condition.
    norm2 = torch.where(norm2 < 1e-20, torch.zeros_like(norm2), torch.ones_like(norm2))
    weighted = quality[:, None] * features
    initial_diagonal = quality.square() * (norm2 + ridge)
    remaining = initial_diagonal.clone()
    coefficients = features.new_zeros((budget, m))
    alive = torch.ones(m, device=features.device, dtype=torch.bool)
    index = torch.arange(m, device=features.device)
    infinity = torch.iinfo(torch.int64).max
    for t in range(min(budget, int(counts.max()))):
        active = counts > t
        maxima = remaining.new_full((number,), -torch.inf).scatter_reduce_(0, groups, remaining, reduce='amax')
        ties = alive & active[groups] & (remaining == maxima[groups])
        best_priority = torch.full((number,), infinity, dtype=torch.long, device=features.device)
        best_priority.scatter_reduce_(0, groups, torch.where(ties, priority, infinity), reduce='amin')
        possible = ties & (priority == best_priority[groups])
        chosen = torch.full((number,), m, dtype=torch.long, device=features.device)
        chosen.scatter_reduce_(0, groups, torch.where(possible, index, m), reduce='amin')
        if not torch.all(chosen[active] < m):
            raise ValueError('A nonempty candidate group has no valid pivot')
        safe = chosen.clamp(max=m - 1)
        pivot = torch.where(active, remaining[safe], torch.ones_like(logdet))
        if not torch.all(pivot > 0) or not torch.isfinite(pivot).all():
            raise FloatingPointError('Nonpositive conditional determinant pivot')
        selected[active, t] = chosen[active]
        pivots[active, t] = pivot[active]
        logdet += torch.log(pivot)
        if diagonal:
            row = torch.zeros_like(remaining)
        else:
            row = (weighted * weighted[safe][groups]).sum(1)
        row[chosen[active]] = initial_diagonal[chosen[active]]
        inner = (coefficients[:t, safe][:, groups] * coefficients[:t]).sum(0)
        increment = (row - inner) / pivot.sqrt()[groups]
        increment = torch.where(active[groups], increment, torch.zeros_like(increment))
        coefficients[t] = increment
        remaining -= increment.square()
        alive[chosen[active]] = False
        remaining[chosen[active]] = -torch.inf
        if not torch.all(remaining[alive] >= -1e-10):
            raise FloatingPointError('Cholesky residual lost positive semidefiniteness')
        remaining[alive] = remaining[alive].clamp_min(0.)
    return dict(indices=selected, counts=counts, logdet=logdet, pivots=pivots)


class DiversityContext:
    def __init__(self, edges, nodes, neural_graph):
        row, col = np.concatenate((edges, edges[::-1]), axis=1)
        keep = row != col
        adjacency = sparse.csr_matrix((np.ones(int(keep.sum()), dtype=np.float32),
                                      (row[keep], col[keep])), shape=(nodes, nodes))
        adjacency.sum_duplicates()
        adjacency.data[:] = 1.
        adjacency.sort_indices()
        self.adjacency, self.neural_graph, self.nodes = adjacency, neural_graph, int(nodes)
        self.items, self.packed = {}, OrderedDict()

    def extract(self, u, v):
        u, v = int(u), int(v)
        if u == v or not (0 <= u < self.nodes and 0 <= v < self.nodes):
            raise ValueError('Expected distinct query endpoints')
        if self.adjacency[u, v] != 0 or self.adjacency[v, u] != 0:
            raise ValueError('Query fact must be removed before neighbor selection')
        key = (u, v)
        if key not in self.items:
            a = self.adjacency
            left = a.indices[a.indptr[u]:a.indptr[u + 1]]
            right = a.indices[a.indptr[v]:a.indptr[v + 1]]
            nodes = np.union1d(left, right)
            nodes = nodes[(nodes != u) & (nodes != v)]
            membership = np.c_[np.isin(nodes, left), np.isin(nodes, right)].astype(np.int64)
            self.items[key] = nodes, membership
        return self.items[key]

    def batch(self, pairs):
        values = pairs.detach().cpu().numpy()
        if values.ndim != 2 or values.shape[0] != 2 or not values.shape[1]:
            raise ValueError('Expected a nonempty pair matrix')
        key = (str(pairs.device), values.tobytes())
        if key not in self.packed:
            all_nodes, memberships, groups = [], [], []
            for i, (u, v) in enumerate(values.T):
                nodes, membership = self.extract(u, v)
                all_nodes.append(nodes)
                memberships.append(membership)
                groups.append(np.full(len(nodes), i, dtype=np.int64))
            self.packed[key] = dict(nodes=torch.as_tensor(np.concatenate(all_nodes), device=pairs.device),
                membership=torch.as_tensor(np.concatenate(memberships), device=pairs.device),
                groups=torch.as_tensor(np.concatenate(groups), device=pairs.device), number=values.shape[1])
            while len(self.packed) > 2:
                self.packed.popitem(last=False)
        self.packed.move_to_end(key)
        return self.packed[key]


class DeterminantalSelectionGraph(nn.Module):
    def __init__(self, dimensions, relations, p, kind):
        super().__init__()
        self.kind, self.p = kind, p
        width = p['width']
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dimensions])
        self.norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([graph_lib.SparseTypedAttention(width, p['heads'],
            2 * relations + 2, p['dropout']) for _ in range(p['depth'])])
        self.evidence = nn.Sequential(nn.Linear(3 * width + 2, width), nn.GELU(), nn.Linear(width, width))
        self.readout = nn.Sequential(nn.Linear(5 * width, width), nn.GELU(),
                                    nn.Dropout(p['dropout']), nn.Linear(width, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()
        self.selection_diagnostics = dict(calls=0, queries=0, candidates=0, selected=0,
            pruned_queries=0, max_candidates=0, minimum_selected_pivot=1.)

    def encode(self, xs, context):
        edges, relation, counts = context.neural_graph
        parts, start = [], 0
        for x, project in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            parts.append(project(torch.cat((x / rms, counts[start:start + len(x)]), dim=-1)))
            start += len(x)
        h = F.gelu(self.norm(torch.cat(parts)))
        for layer in self.layers:
            h = layer(h, edges, relation)
        return h

    @torch.no_grad()
    def select(self, h, pairs, packed):
        unit = normalize_rows(h.detach().double())
        query = normalize_rows((h[pairs[0]] + h[pairs[1]]).detach().double())
        feature = unit[packed['nodes']]
        quality = conditional_quality(feature, query, packed['groups'], packed['number'], self.p['quality_floor'])
        if self.kind == 'no_condition':
            quality = torch.ones_like(quality)
        # Hidden-feature order, with side membership as the secondary key. Exact
        # duplicate keys have identical evidence values; node IDs only choose an
        # interchangeable representative in that last case.
        _, inverse = np.unique(h.detach().cpu().numpy(), axis=0, return_inverse=True)
        ranks = torch.as_tensor(inverse, device=h.device)
        membership = packed['membership']
        priority = 4 * ranks[packed['nodes']] + 2 * membership[:, 0] + membership[:, 1]
        return greedy_subset(feature, quality, packed['groups'], priority, packed['number'],
            self.p['retained_neighbors'], self.p['gram_diagonal_ridge'], diagonal=self.kind == 'no_diversity')

    def forward(self, xs, context, pairs):
        h = self.encode(xs, context)
        packed = context.batch(pairs)
        selected = self.select(h, pairs, packed)
        indices, counts = selected['indices'], selected['counts']
        live = indices >= 0
        left, right = h[pairs[0]], h[pairs[1]]
        if len(packed['nodes']):
            safe = indices.clamp_min(0)
            value = h[packed['nodes'][safe]]
            membership = packed['membership'][safe].to(h.dtype)
            evidence = self.evidence(torch.cat((value, value * left[:, None], value * right[:, None], membership), -1))
            pooled = (evidence * live[:, :, None]).sum(1) / live.sum(1).clamp_min(1)[:, None]
        else:
            pooled = h.new_zeros((pairs.shape[1], h.shape[1]))
        d = self.selection_diagnostics
        d['calls'] += 1
        d['queries'] += pairs.shape[1]
        d['candidates'] += len(packed['nodes'])
        d['selected'] += int(live.sum())
        d['pruned_queries'] += int((counts > self.p['retained_neighbors']).sum())
        d['max_candidates'] = max(d['max_candidates'], int(counts.max()))
        if live.any():
            d['minimum_selected_pivot'] = min(d['minimum_selected_pivot'], float(selected['pivots'][live].min()))
        return self.readout(torch.cat((left, right, left * right, (left - right).abs(), pooled), -1)).squeeze(-1)


def setup(a, p, kind, device):
    if set(a) - TRAIN_KEYS:
        raise ValueError('Determinantal selection accepts training arrays only')
    if kind not in KINDS:
        raise ValueError('Unregistered determinant arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        return ordinary.IdentityResidual().to(device), [], [None] * p['folds'], folds
    n, relations = int(a['node_counts'].sum()), int(a['relation'].max()) + 1
    graphs = []
    for f in range(p['folds']):
        edges, labels = graph_lib.masked_physical(a, np.flatnonzero(a['folds'] == f))
        neural = graph_lib.graph_inputs(edges, labels, np.empty((2, 0), dtype=np.int64), n, relations, device)
        graphs.append(DiversityContext(edges, n, neural))
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = DeterminantalSelectionGraph([x.shape[1] for x in xs], relations, p, kind).to(device)
    return model, xs, graphs, folds
