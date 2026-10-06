"""CORE-inspired completion and query-specific information-bottleneck pruning.

Independent implementation of arXiv:2404.11032v1, Sections 3 and Appendix B.
Completion is restricted to same-type virtual edges. This is an explicit local
adaptation: it cannot insert a target RNA-drug query or a biological fact.
"""
from collections import OrderedDict
import hashlib
import math

import numpy as np
from scipy import sparse
from scipy.stats import rankdata
import torch
from torch import nn
from torch.nn import functional as F

from improve import bicentric_subgraph as local
from improve import overlap_block_models as physical
from improve import sharpness_graph_v1 as shared

KINDS = ('full', 'no_kl', 'no_completion')
_CONTEXTS = OrderedDict()
identity_task = shared.identity_task


def parameters():
    return dict(width=32, depth=2, dropout=.1, folds=5, query_chunk=128,
        completion_ratio=.1, original_prior=.8, virtual_prior=.5,
        information_weight=.001, temperature=.67,
        lnc_policy='identity_strong_reference')


def boolean_adjacency(edges, nodes):
    rows, columns = np.concatenate([edges, edges[::-1]], axis=1)
    keep = rows != columns
    graph = sparse.csr_matrix((np.ones(int(keep.sum()), dtype=np.float64),
                               (rows[keep], columns[keep])), shape=(nodes, nodes))
    graph.sum_duplicates()
    graph.data[:] = 1.
    graph.sort_indices()
    return graph


def complete_same_type(edges, node_counts, ratio):
    """Resource-allocation proposals; ties at the budget boundary are all kept."""
    if not 0 <= ratio <= 1:
        raise ValueError('Invalid completion ratio')
    counts = np.asarray(node_counts, dtype=np.int64)
    graph = boolean_adjacency(edges, int(counts.sum()))
    types = np.repeat(np.arange(len(counts)), counts)
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    proposal = sparse.triu(graph @ sparse.diags(1 / np.maximum(degree, 1)) @ graph, k=1).tocoo()
    known = np.asarray(graph[proposal.row, proposal.col]).reshape(-1)
    eligible = ((types[proposal.row] == types[proposal.col]) & (known == 0) & (proposal.data > 0))
    rows, columns, values = proposal.row[eligible], proposal.col[eligible], proposal.data[eligible]
    budget = min(int(math.floor(ratio * (graph.nnz // 2))), len(values))
    cutoff = None
    if budget:
        cutoff = float(np.partition(values, len(values) - budget)[len(values) - budget])
        chosen = values >= cutoff
        rows, columns, values = rows[chosen], columns[chosen], values[chosen]
        ranks = rankdata(-values, method='average')
        bins = 1 + np.minimum(9, np.floor(10 * (ranks - 1) / len(values)).astype(np.int64))
    else:
        rows, columns, values = rows[:0], columns[:0], values[:0]
        bins = np.empty(0, dtype=np.int64)
    origin = sparse.csr_matrix((np.r_[bins, bins], (np.r_[rows, columns], np.r_[columns, rows])),
                               shape=graph.shape, dtype=np.int64)
    augmented = graph + (origin != 0).astype(np.float64)
    augmented.sort_indices()
    upper = sparse.triu(augmented, k=1).tocoo()
    audit = dict(original_undirected_edges=graph.nnz // 2,
        eligible_same_type_proposals=int(eligible.sum()), nominal_addition_budget=budget,
        virtual_undirected_edges=len(values), cutoff=cutoff, tied_cutoff_edges_retained=True,
        cross_type_virtual_edges=0, target_queries_cannot_be_virtual_edges=True,
        virtual_edges_are_supervision=False, completion_uses_labels=False,
        completion_input='Current support-only physical graph; no held-out target facts.')
    return np.stack([upper.row, upper.col]).astype(np.int64), origin, audit


class BottleneckContext(local.SubgraphContext):
    def __init__(self, edges, node_counts, ratio):
        augmented, self.origin, self.audit = complete_same_type(edges, node_counts, ratio)
        super().__init__(augmented, int(np.sum(node_counts)))
        self.extra = {}

    def extract_extra(self, u, v):
        key = (int(u), int(v))
        if key in self.extra:
            return self.extra[key]
        nodes, labels, edge, weights = self.extract(u, v)
        source, target = edge
        unique = edge[:, source < target]
        codes = unique[0] * len(nodes) + unique[1]
        direction_codes = np.minimum(source, target) * len(nodes) + np.maximum(source, target)
        groups = np.zeros(len(source), dtype=np.int64)
        nonself = source != target
        groups[nonself] = np.searchsorted(codes, direction_codes[nonself]) + 1
        if len(codes):
            np.testing.assert_array_equal(codes[groups[nonself] - 1], direction_codes[nonself])
        origins = np.asarray(self.origin[nodes[unique[0]], nodes[unique[1]]]).reshape(-1)
        result = (nodes, labels, edge, weights, unique, groups, origins)
        self.extra[key] = result
        return result

    def batch(self, pairs):
        array = pairs.detach().cpu().numpy()
        if array.ndim != 2 or array.shape[0] != 2 or array.shape[1] == 0:
            raise ValueError('Expected a nonempty pair matrix')
        key = (str(pairs.device), array.tobytes())
        if key in self.packed:
            self.packed.move_to_end(key)
            return self.packed[key]
        fields = {name: [] for name in ('nodes', 'labels', 'edges', 'weights', 'unique',
                                        'groups', 'origins', 'node_batch', 'edge_batch')}
        starts, node_offset, edge_offset = [], 0, 0
        for i, (u, v) in enumerate(array.T):
            nodes, labels, edges, weights, unique, groups, origins = self.extract_extra(int(u), int(v))
            starts.append(node_offset)
            fields['nodes'].append(nodes)
            fields['labels'].append(labels)
            fields['edges'].append(edges + node_offset)
            fields['weights'].append(weights)
            fields['unique'].append(unique + node_offset)
            fields['groups'].append(np.where(groups > 0, groups + edge_offset, 0))
            fields['origins'].append(origins)
            fields['node_batch'].append(np.full(len(nodes), i, dtype=np.int64))
            fields['edge_batch'].append(np.full(len(origins), i, dtype=np.int64))
            node_offset += len(nodes)
            edge_offset += len(origins)
        result = {name: torch.as_tensor(np.concatenate(values, axis=1 if name in ('edges', 'unique') else 0),
                                        device=pairs.device) for name, values in fields.items()}
        result.update(starts=torch.as_tensor(starts, device=pairs.device), count=len(starts))
        result['node_count'] = torch.bincount(result['node_batch'], minlength=result['count'])
        self.packed[key] = result
        while len(self.packed) > 2:
            self.packed.popitem(last=False)
        return result


def context(a, omitted, p, kind):
    edges, _ = physical.masked_physical(a, omitted)
    ratio = 0. if kind == 'no_completion' else p['completion_ratio']
    key = (hashlib.sha256(np.ascontiguousarray(edges).tobytes()).hexdigest(),
           tuple(a['node_counts']), float(ratio))
    if key not in _CONTEXTS:
        _CONTEXTS[key] = BottleneckContext(edges, a['node_counts'], ratio)
    _CONTEXTS.move_to_end(key)
    while len(_CONTEXTS) > 10:
        _CONTEXTS.popitem(last=False)
    return _CONTEXTS[key]


def bernoulli_kl(logits, prior):
    """Analytic KL per undirected Bernoulli edge, not per directed message."""
    if torch.any((prior <= 0) | (prior >= 1)):
        raise ValueError('Prior must be strictly inside (0, 1)')
    probability = logits.sigmoid()
    return probability * (F.logsigmoid(logits) - prior.log()) + (
        1 - probability) * (F.logsigmoid(-logits) - torch.log1p(-prior))


def relaxed_bernoulli(logits, temperature, generator):
    if generator is None or temperature <= 0:
        raise ValueError('Training requires a private generator and positive temperature')
    uniform = torch.rand(logits.shape, device=logits.device, dtype=logits.dtype,
                         generator=generator).clamp(1e-6, 1 - 1e-6)
    return ((logits + uniform.log() - torch.log1p(-uniform)) / temperature).sigmoid()


class BottleneckGraph(nn.Module):
    def __init__(self, dimensions, p, kind):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unknown information-bottleneck arm')
        self.kind, self.p = kind, dict(p)
        width = p['width']
        self.project = nn.ModuleList([nn.Linear(d, width) for d in dimensions])
        self.mark = nn.Embedding(5, width)
        self.input_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([nn.Linear(2 * width, width) for _ in range(p['depth'])])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(p['depth'])])
        self.origin = nn.Embedding(11, 8)
        self.query = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
        self.key = nn.Sequential(nn.Linear(2 * width + 8, width), nn.ReLU(), nn.Linear(width, width))
        self.readout = nn.Sequential(nn.Linear(4 * width, 64), nn.ReLU(),
                                     nn.Dropout(p['dropout']), nn.Linear(64, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()

    @staticmethod
    def pool(h, packed):
        sums = h.new_zeros((packed['count'], h.shape[1])).index_add(0, packed['node_batch'], h)
        return sums / packed['node_count'][:, None].to(h.dtype)

    def encode(self, initial, packed, masks=None):
        source, target = packed['edges']
        weights = packed['weights']
        if masks is not None:
            # Fixed inflated-graph normalization; sampled values restrict messages.
            weights = weights * torch.cat((masks.new_ones(1), masks))[packed['groups']]
        h = initial
        for linear, norm in zip(self.layers, self.norms):
            aggregate = h.new_zeros(h.shape).index_add(0, target, h[source] * weights[:, None])
            h = F.relu(norm(linear(torch.cat((h, aggregate), dim=-1))))
            h = F.dropout(h, self.p['dropout'], self.training)
        return h

    def forward_with_kl(self, xs, graph, pairs, generator=None):
        packed = graph.batch(pairs)
        base = torch.cat([layer(x / x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt())
                          for layer, x in zip(self.project, xs)])
        initial = self.input_norm(base[packed['nodes']] + self.mark(packed['labels']))
        preliminary = self.encode(initial, packed)
        pooled = self.pool(preliminary, packed)
        left, right = packed['unique']
        edge = torch.cat((preliminary[left] + preliminary[right],
                          (preliminary[left] - preliminary[right]).abs(),
                          self.origin(packed['origins'])), dim=-1)
        logits = (self.query(pooled)[packed['edge_batch']] * self.key(edge)).sum(-1) / math.sqrt(self.p['width'])
        probability = logits.sigmoid()
        if self.training:
            mask = relaxed_bernoulli(logits, self.p['temperature'], generator)
        else:
            # Probability-weighted inference is a plug-in approximation, not exact marginalization.
            mask = probability
        final = self.encode(initial, packed, mask)
        lhs, rhs = final[packed['starts']], final[packed['starts'] + 1]
        value = self.readout(torch.cat((lhs, rhs, lhs * rhs, self.pool(final, packed)), dim=-1)).squeeze(-1)
        prior = torch.where(packed['origins'] == 0,
                            logits.new_tensor(self.p['original_prior']), logits.new_tensor(self.p['virtual_prior']))
        edge_kl = bernoulli_kl(logits, prior)
        # Sum within each subgraph and average subgraphs in the executor (paper Appendix A.1).
        kl = logits.new_zeros(packed['count']).index_add(0, packed['edge_batch'], edge_kl)
        detail = dict(undirected_edges=len(logits), graphs=packed['count'],
            mean_retention=0. if not len(logits) else float(probability.detach().mean()),
            virtual_edges=int((packed['origins'] > 0).sum()))
        return value, kl, detail

    def forward(self, xs, graph, pairs):
        if self.training:
            raise ValueError('Use forward_with_kl with a private generator during fitting')
        return self.forward_with_kl(xs, graph, pairs)[0]


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Unregistered information-bottleneck arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == fold), device=device) for fold in range(p['folds'])]
    if identity_task(a, p):
        return shared.IdentityResidual().to(device), [], [None] * p['folds'], folds
    graphs = [context(a, np.flatnonzero(a['folds'] == fold), p, kind) for fold in range(p['folds'])]
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = BottleneckGraph([x.shape[1] for x in xs], p, kind).to(device)
    return model, xs, graphs, folds
