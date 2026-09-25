"""Auxiliary-graph attention kernels with a direct reference-offset likelihood.

All MiDrug target edges are absent from the covariance. The fixed kernel is
therefore independent of every MiDrug training label. Convex kernel logistic
regression learns the reference residual without feeding target labels into
another training row's representation. The result is a MAP function, not a
marginal Bernoulli GP prediction.
"""
import numpy as np
import torch
from torch.nn import functional as F

from improve import attention_gp_rank_v1 as covariance

TRAIN_KEYS = covariance.TRAIN_KEYS
KINDS = covariance.KINDS


def auxiliary_arrays(a):
    if set(a) - TRAIN_KEYS:
        raise ValueError('Training-only array contract violated')
    keep = a['target_codes'] < 0
    return {key: (value[:, keep] if key == 'edges' else value[keep]
                  if key in ('target_codes', 'relation') else value)
            for key, value in a.items()}


def entity_kernels(a, kind, depth=2, device='cpu'):
    return covariance.entity_kernels(auxiliary_arrays(a), np.array([], dtype=np.int64), kind, depth, device)


def objective(kernel, alpha, y, offset, precision):
    value = kernel @ alpha
    return F.binary_cross_entropy_with_logits(offset+value, y, reduction='sum') + .5*precision*(alpha @ value)


@torch.no_grad()
def solve(kernel, y, offset, precision=10., tolerance=1e-7, max_iterations=50):
    """Damped Newton / GPML-style kernel logistic solve without a kernel inverse."""
    n = len(y)
    if (kernel.shape != (n, n) or offset.shape != y.shape or y.ndim != 1
            or not torch.isfinite(kernel).all() or not torch.isfinite(offset).all()
            or not torch.all((y == 0) | (y == 1)) or precision <= 0):
        raise ValueError('Invalid fixed covariance, offset, or binary training labels')
    alpha = torch.zeros_like(y)
    trace = []
    for iteration in range(max_iterations):
        value = kernel @ alpha
        probability = torch.sigmoid(offset + value)
        residual = y-probability-precision*alpha
        maximum = float(residual.abs().max())
        loss = float(objective(kernel, alpha, y, offset, precision))
        trace.append(dict(iteration=iteration, objective=loss, maximum_dual_residual=maximum))
        if maximum <= tolerance:
            break
        curvature = (probability*(1.-probability)).clamp_min(1e-12)
        root = curvature.sqrt()
        b = curvature*value + y-probability
        kb = kernel @ b / precision
        system = (root[:, None] * kernel * root[None, :]) / precision
        system.diagonal().add_(1.)
        factor = torch.linalg.cholesky(system)
        candidate = (b-root*torch.cholesky_solve((root*kb)[:, None], factor).flatten()) / precision
        direction = candidate-alpha
        for backtrack in range(25):
            trial = alpha + (2.**(-backtrack))*direction
            trial_loss = float(objective(kernel, trial, y, offset, precision))
            if trial_loss <= loss + 1e-10:
                alpha = trial
                trace[-1]['step'] = 2.**(-backtrack)
                break
        else:
            raise ValueError('Kernel logistic Newton line search failed')
    else:
        raise ValueError('Kernel logistic solver did not converge')
    return alpha, trace


@torch.no_grad()
def fit(rna_kernel, drug_kernel, pairs, y, offset, precision=10.):
    device = rna_kernel.device
    pairs = torch.as_tensor(pairs, dtype=torch.int64, device=device)
    y = torch.as_tensor(y, dtype=torch.float64, device=device)
    offset = torch.as_tensor(offset, dtype=torch.float64, device=device)
    r, d = pairs
    kernel = rna_kernel[r[:, None], r[None, :]] * drug_kernel[d[:, None], d[None, :]]
    alpha, trace = solve(kernel, y, offset, precision)
    weights = torch.zeros((len(rna_kernel), len(drug_kernel)), device=device, dtype=torch.float64)
    weights.index_put_((r, d), alpha, accumulate=True)
    mean = rna_kernel @ weights @ drug_kernel.T
    value = kernel @ alpha
    replay_error = float((mean[r, d] - value).abs().max())
    if replay_error > 1e-7:
        raise ValueError('Full-universe matrix prediction disagrees with the training kernel')
    return mean, alpha, trace, replay_error
