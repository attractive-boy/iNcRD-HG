"""Cell-subset sensitivity, reproducible edges, and selective consistency.

Reliabilities are descriptive stability scores under random cell-line subsets.
They are neither calibrated probabilities nor cancer-specific causal effects.
All transformations are label-free and use a previously training-fitted PCA.
"""
import numpy as np
from scipy.stats import kendalltau
import torch
import torch.nn.functional as F


def complementary_partitions(n_features, repeats=8, seed=20260909):
    if n_features < 4 or repeats < 1:
        raise ValueError('At least four cell lines and one partition are required')
    rng = np.random.default_rng(seed)
    return [tuple(np.split(rng.permutation(n_features), [n_features // 2]))
            for _ in range(repeats)]


def expression_views(raw, parameters, partitions):
    """Estimate full PCA scores from complementary cell-line subsets.

    Inverse sampling fractions correct each partial projection. The weighted
    average of two complementary views is exactly the full projection (up to
    floating-point error), including for an odd number of cell lines. Thus an
    average view adds no information; disagreement is the measured quantity.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError('Expected a finite node-by-cell-line expression matrix')
    scale = np.asarray(parameters['input_scale'])
    pc_scale = np.asarray(parameters['component_scale'])
    if np.any(scale <= 0) or np.any(pc_scale <= 0):
        raise ValueError('PCA scales must be positive')
    standardized = (raw - parameters['input_mean']) / scale - parameters['pca_mean']
    components = np.asarray(parameters['pca_components'])
    pc_mean = np.asarray(parameters['component_mean'])
    full = (standardized @ components.T - pc_mean) / pc_scale
    n = raw.shape[1]
    views = []
    for left, right in partitions:
        if not np.array_equal(np.sort(np.concatenate([left, right])), np.arange(n)):
            raise ValueError('Views must be disjoint and cover every cell line exactly once')
        pair = []
        for ids in (left, right):
            if len(ids) == 0:
                raise ValueError('An empty cell-line view is invalid')
            projected = standardized[:, ids] @ components[:, ids].T * (n / len(ids))
            pair.append((projected - pc_mean) / pc_scale)
        np.testing.assert_allclose((len(left) * pair[0] + len(right) * pair[1]) / n,
                                   full, rtol=1e-10, atol=1e-10)
        views.extend(pair)
    views = np.stack(views)
    sensitivity = np.mean((views - full[None]) ** 2, axis=(0, 2))
    reliability = 1.0 / (1.0 + sensitivity)
    return (full.astype(np.float32), views.astype(np.float32),
            reliability.astype(np.float32), sensitivity.astype(np.float32))


def canonical_edges(edge_index):
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError('Expected a 2 by E edge array')
    nonself = edge_index[0] != edge_index[1]
    pairs, inverse = np.unique(np.sort(edge_index[:, nonself].T, axis=1),
                               axis=0, return_inverse=True)
    return nonself, pairs, inverse


def edge_reliability(raw, edge_index, supplied_similarity, partitions):
    """Require both complementary contexts to support an existing edge.

    Each half is evaluated with Kendall tau-b, matching the supplied metric.
    Negative correlations are oriented by the ORIGINAL sign when measuring
    support. Propagation remains positive, so this isolates reliability and
    makes no biological inhibition or signed-message claim.
    """
    nonself, pairs, inverse = canonical_edges(edge_index)
    original = supplied_similarity[pairs[:, 0], pairs[:, 1]]
    if not np.isfinite(original).all() or np.any(original == 0):
        raise ValueError('Existing non-self edges need finite nonzero similarities')
    orientation = np.sign(original)
    supports, agreements = [], []
    for b, (left, right) in enumerate(partitions):
        correlations = []
        for ids in (left, right):
            block = raw[:, ids]
            values = np.array([kendalltau(block[i], block[j]).statistic
                               for i, j in pairs], dtype=np.float64)
            correlations.append(np.nan_to_num(values, nan=0.0))
        a, c = orientation * correlations[0], orientation * correlations[1]
        supports.append(np.maximum(np.minimum(a, c), 0.0))
        agreements.append((a > 0) & (c > 0))
        print('edge reliability partition', b + 1, '/', len(partitions),
              'undirected pairs', len(pairs), flush=True)
    support = np.mean(supports, axis=0)
    # The fixed 0.05 floor prevents a reliability estimate from deleting edges.
    factor = np.clip(support / np.abs(original), 0.05, 1.0)
    per_edge = np.ones(edge_index.shape[1], dtype=np.float32)
    per_edge[nonself] = factor[inverse]
    return per_edge, {
        'pairs': pairs, 'original': original, 'support': support,
        'factor': factor, 'sign_agreement': np.mean(agreements, axis=0),
    }


def normalize_incoming(edge_index, strength, n_nodes):
    """Preserve every self loop and the total mass of non-self messages."""
    edge_index = np.asarray(edge_index, dtype=np.int64)
    strength = np.asarray(strength, dtype=np.float64)
    if strength.shape != (edge_index.shape[1],) or not np.isfinite(strength).all():
        raise ValueError('Expected one finite strength per edge')
    if np.any(strength <= 0):
        raise ValueError('Multiplicative strengths must be positive')
    active = edge_index[0] != edge_index[1]
    dst = edge_index[1, active]
    degree = np.bincount(dst, minlength=n_nodes)
    total = np.bincount(dst, weights=strength[active], minlength=n_nodes)
    weights = np.ones_like(strength)
    weights[active] = strength[active] * degree[dst] / total[dst]
    np.testing.assert_allclose(np.bincount(dst, weights=weights[active], minlength=n_nodes),
                               degree, rtol=1e-12, atol=1e-12)
    return weights.astype(np.float32)


def shuffle_node_reliability(reliability, fit_ids, seed=20260909):
    """Permute separately inside fitted and remaining nodes, without labels."""
    reliability = np.asarray(reliability)
    fit_ids = np.asarray(fit_ids, dtype=np.int64)
    others = np.setdiff1d(np.arange(len(reliability)), fit_ids)
    rng = np.random.default_rng(seed)
    shuffled = reliability.copy()
    for ids in (fit_ids, others):
        shuffled[ids] = reliability[rng.permutation(ids)]
    return shuffled


def shuffle_edge_reliability(edge_index, supplied_similarity, factors, seed=20260909):
    """Shuffle undirected edge factors within sign and |similarity| deciles.

    This retains the coarse strength/reliability relationship, the factor
    distribution, and reciprocity before destination normalization.
    """
    active, pairs, inverse = canonical_edges(edge_index)
    similarity = supplied_similarity[pairs[:, 0], pairs[:, 1]]
    _, representative = np.unique(inverse, return_index=True)
    original = np.asarray(factors)[active][representative]
    bins = np.searchsorted(np.quantile(np.abs(similarity), np.linspace(0, 1, 11))[1:-1],
                           np.abs(similarity), side='right')
    rng = np.random.default_rng(seed)
    shuffled = original.copy()
    for sign in (-1, 1):
        for bin_id in range(10):
            ids = np.flatnonzero((np.sign(similarity) == sign) & (bins == bin_id))
            shuffled[ids] = original[rng.permutation(ids)]
    out = np.ones_like(factors)
    out[active] = shuffled[inverse]
    return out


def consistency_loss(student_logits, teacher_logits, reliability=None):
    """Bernoulli KL from a detached full-expression teacher to one cell view.

    Mean-one weights match the total regularization strength of the uniform
    control. No node is treated as a contrastive negative for another node.
    """
    teacher = teacher_logits.detach()
    probabilities = teacher.sigmoid()
    divergence = (F.binary_cross_entropy_with_logits(student_logits, probabilities, reduction='none')
                  - F.binary_cross_entropy_with_logits(teacher, probabilities, reduction='none'))
    if reliability is None:
        return divergence.mean()
    weights = reliability.detach()
    if weights.shape != divergence.shape or not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise ValueError('Consistency requires one finite positive reliability per pair')
    weights = weights / weights.mean()
    return (weights * divergence).mean()
