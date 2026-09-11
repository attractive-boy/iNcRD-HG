"""Marginal/interaction decoding and query-absent pair-exchange supervision.

Unknown crossed pairs are ranking alternatives, not experimentally verified
negative or drug-sensitive associations. A four-score contrast cancels any
additive RNA-only and drug-only score; it does not identify a causal effect.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DecomposedDecoder(nn.Module):
    """Two endpoint functions plus a centered low-rank interaction surface."""
    def __init__(self, hidden, rna_type, fit_rna, fit_drug, rank=32):
        super().__init__()
        if rank < 1 or len(fit_rna) == 0 or len(fit_drug) == 0:
            raise ValueError('Positive rank and nonempty centering populations required')
        self.rna_type, self.rank = rna_type, rank
        self.rna_marginal = nn.Sequential(nn.Linear(hidden, hidden), nn.LeakyReLU(.1), nn.Linear(hidden, 1))
        self.drug_marginal = nn.Sequential(nn.Linear(hidden, hidden), nn.LeakyReLU(.1), nn.Linear(hidden, 1))
        self.rna_projection = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, rank, bias=False))
        self.drug_projection = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, rank, bias=False))
        self.register_buffer('fit_rna', torch.as_tensor(fit_rna, dtype=torch.long))
        self.register_buffer('fit_drug', torch.as_tensor(fit_drug, dtype=torch.long))

    def tables(self, z):
        r, d = z[self.rna_type], z['drug']
        u, v = self.rna_projection(r), self.drug_projection(d)
        u = u - u[self.fit_rna].mean(0, keepdim=True)
        v = v - v[self.fit_drug].mean(0, keepdim=True)
        return self.rna_marginal(r).flatten(), self.drug_marginal(d).flatten(), u, v

    def components(self, z, pairs):
        a, b, u, v = self.tables(z)
        r, d = pairs
        marginal = a[r] + b[d]
        interaction = (u[r] * v[d]).sum(-1) / math.sqrt(self.rank)
        return marginal, interaction

    def forward(self, z, pairs):
        marginal, interaction = self.components(z, pairs)
        return marginal + interaction


def sample_queries(labels, fraction, seed):
    """Stratified query rows using a dedicated RNG shared by all masked arms."""
    labels = np.asarray(labels)
    if not 0 < fraction <= 1 or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Need two classes and a fraction in (0, 1]')
    rng = np.random.default_rng(seed)
    chosen = []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        count = max(1, int(round(len(indices) * fraction)))
        chosen.append(rng.choice(indices, count, replace=False))
    return np.sort(np.concatenate(chosen))


def positive_matrix(pairs, labels, n_rna, n_drug):
    positive = np.zeros((n_rna, n_drug), dtype=bool)
    positive[tuple(pairs[:, np.asarray(labels) == 1])] = True
    return positive


def sample_exchanges(query_positives, known_positive, size, seed):
    """Return (r, d, s, e), contrasting (r,d),(s,e) with (r,e),(s,d).

    Both source positives belong to the query set, and neither crossed pair
    may be a fitting-set positive. No held-out labels or exclusion set enters
    this API. Repeated sampled exchanges are allowed; attempts are bounded.
    """
    p = np.asarray(query_positives, dtype=np.int64)
    if p.ndim != 2 or p.shape[0] != 2 or size < 0:
        raise ValueError('Invalid query positives or sample count')
    if p.shape[1] < 2 or size == 0:
        return np.empty((4, 0), dtype=np.int64)
    if not known_positive[tuple(p)].all():
        raise ValueError('Query positives are missing from fitting positives')
    rng = np.random.default_rng(seed)
    blocks, count = [], 0
    for _ in range(16):
        candidates = max(64, 3 * (size - count))
        first, second = rng.integers(p.shape[1], size=(2, candidates))
        r, d = p[:, first]
        s, e = p[:, second]
        keep = (r != s) & (d != e) & ~known_positive[r, e] & ~known_positive[s, d]
        block = np.stack([r[keep], d[keep], s[keep], e[keep]])[:, :size-count]
        if block.shape[1]:
            blocks.append(block)
            count += block.shape[1]
        if count == size:
            break
    return np.concatenate(blocks, axis=1) if blocks else np.empty((4, 0), dtype=np.int64)


def exchange_pairs(exchanges):
    r, d, s, e = exchanges
    return np.stack([np.concatenate([r, s, r, s]), np.concatenate([d, e, e, d])])


def auxiliary_loss(scores, mode):
    """Same four scored pairs in the exchange and ordinary ranking controls."""
    if scores.ndim != 2 or scores.shape[0] != 4:
        raise ValueError('Expected four rows of exchange scores')
    if mode not in ('exchange', 'pairwise'):
        raise ValueError(mode)
    if scores.shape[1] == 0:
        return scores.sum() * 0
    p, q, u, v = scores
    if mode == 'exchange':
        return F.softplus(-.5 * (p + q - u - v)).mean()
    differences = torch.stack([p-u, p-v, q-u, q-v])
    return F.softplus(-differences).mean()


def remove_target_pairs(edges, weights, task, rna_type, n_drug, held_pairs):
    """Delete queried positives in BOTH directions without touching other edges."""
    key, reverse = (rna_type, task, 'drug'), ('drug', 'rev_' + task, rna_type)
    if key not in edges or reverse not in edges:
        raise ValueError('Both target relation directions must be present')
    held = torch.as_tensor(held_pairs, dtype=torch.long, device=edges[key].device)
    forbidden = held[0] * n_drug + held[1]
    result_edges, result_weights = dict(edges), dict(weights)
    for relation in (key, reverse):
        index = edges[relation] if relation == key else edges[relation].flip(0)
        keep = ~torch.isin(index[0] * n_drug + index[1], forbidden)
        result_edges[relation] = edges[relation][:, keep]
        result_weights[relation] = weights[relation][keep]
    return result_edges, result_weights


def restrict_training_graph(graph, weights, task, rna_type, pairs, labels):
    """Build a support graph from the fitting labels, excluding inner holdouts."""
    graph = graph.clone()
    key, reverse = (rna_type, task, 'drug'), ('drug', 'rev_' + task, rna_type)
    n_drug = graph['drug'].num_nodes
    positive = pairs[:, labels == 1]
    allowed = positive[0] * n_drug + positive[1]
    result_weights = dict(weights)
    for relation in (key, reverse):
        index = graph[relation].edge_index.numpy()
        index = index if relation == key else index[::-1]
        keep = np.isin(index[0] * n_drug + index[1], allowed)
        graph[relation].edge_index = graph[relation].edge_index[:, keep]
        result_weights[relation] = weights[relation][keep]
    actual = graph[key].edge_index.numpy()
    if set(map(tuple, actual.T)) != set(map(tuple, positive.T)):
        raise ValueError('Fitting positives disagree with support graph')
    if set(map(tuple, graph[reverse].edge_index.flip(0).numpy().T)) != set(map(tuple, positive.T)):
        raise ValueError('Reverse support graph differs')
    graph[key].edge_label_index = torch.from_numpy(pairs.copy())
    graph[key].edge_label = torch.from_numpy(labels.copy())
    return graph, result_weights


def exchange_support(pairs, labels, n_rna, n_drug):
    """Count exact observed-negative tetrads to document sampling sparsity."""
    positive = positive_matrix(pairs, labels, n_rna, n_drug)
    negative = np.zeros_like(positive)
    negative[tuple(pairs[:, labels == 0])] = True
    p = pairs[:, labels == 1]
    legal_count, observed_count = 0, 0
    covered = np.zeros(p.shape[1], dtype=bool)
    for first in range(p.shape[1]):
        r, d = p[:, first]
        s, e = p[:, first+1:]
        legal = (r != s) & (d != e) & ~positive[r, e] & ~positive[s, d]
        observed = legal & negative[r, e] & negative[s, d]
        legal_count += int(legal.sum())
        observed_count += int(observed.sum())
        if observed.any():
            covered[first] = True
            covered[first+1:][observed] = True
    return dict(positive_edges=p.shape[1], legal_unlabeled_swaps=legal_count,
                observed_negative_swaps=observed_count, covered_positive_edges=int(covered.sum()))
