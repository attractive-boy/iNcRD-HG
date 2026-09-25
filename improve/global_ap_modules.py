"""A smooth global average-precision objective for an ensemble's predictions.

Each positive anchor compares against the entire fitting candidate pool.
Uniformly sampled positive anchors estimate the outer average only; negatives
are never restricted to an individual drug or to a training minibatch.
"""
import numpy as np
import torch


def positive_anchors(labels, budget, seed):
    if budget < 1:
        raise ValueError('The positive-anchor budget must be positive')
    y = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    if y.ndim != 1 or not np.isin(y, (0, 1)).all():
        raise ValueError('Expected binary labels')
    indices = np.flatnonzero(y == 1)
    if len(indices) > budget:
        indices = np.random.RandomState(seed % (2**32-1)).choice(indices, budget, replace=False)
    return np.sort(indices)


def smooth_ap_loss(scores, labels, anchors, temperature=.05):
    """1 - mean soft precision at each positive's soft rank.

    R_i = 1 + sum_{j != i} sigmoid((s_j-s_i)/temperature).
    R_i+ uses the same sum over fitting positives. AP ~= mean_i(R_i+/R_i).
    Self-comparisons are removed exactly. Ties have a soft-rank interpretation;
    this surrogate is not the exact sklearn tie-handling convention.
    """
    if scores.ndim != 1 or labels.shape != scores.shape or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Invalid scores, labels or temperature')
    if not bool(torch.isfinite(scores).all()) or not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError('Expected finite scores and binary labels')
    anchors = torch.as_tensor(anchors, dtype=torch.long, device=scores.device)
    if anchors.ndim != 1:
        raise ValueError('Expected a vector of positive anchor indices')
    if anchors.numel() == 0:
        if bool((labels == 1).any()):
            raise ValueError('No positive anchors were provided')
        return scores.sum()*0.
    if bool((anchors < 0).any()) or bool((anchors >= len(scores)).any()):
        raise ValueError('Anchor index out of range')
    if anchors.unique().numel() != anchors.numel() or not bool((labels[anchors] == 1).all()):
        raise ValueError('Anchors must be distinct fitting positives')
    comparisons = torch.sigmoid((scores[None, :] - scores[anchors, None]) / temperature)
    # sigmoid(0) is exactly 0.5; replace the self term by the leading rank 1.
    rank = .5 + comparisons.sum(1)
    positive_rank = .5 + (comparisons * labels[None, :]).sum(1)
    return 1. - (positive_rank / rank).mean()
