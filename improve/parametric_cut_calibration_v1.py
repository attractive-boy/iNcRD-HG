"""Training-only scalar calibration for a cross-fitted graph-cut rank.

This supplies a common residual interface, not the new mechanism. The graph
construction and the two-stage cut objective live in parametric_cut_rank_v1.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def objective(theta, x, y, offset, l2):
    design = np.column_stack((x, np.ones(len(x))))
    logits = offset + design @ theta
    value = np.mean(np.logaddexp(0., logits) - y * logits) + .5 * l2 * (theta @ theta)
    gradient = design.T @ (expit(logits) - y) / len(y) + l2 * theta
    return float(value), gradient


def fit(rank_oof, y_train, anchor_oof, l2=.01):
    rank_oof, y_train, anchor_oof = [np.asarray(x, dtype=np.float64)
                                    for x in (rank_oof, y_train, anchor_oof)]
    if (rank_oof.ndim != 1 or not len(rank_oof)
            or not (rank_oof.shape == y_train.shape == anchor_oof.shape)
            or not all(np.isfinite(x).all() for x in (rank_oof, y_train, anchor_oof))
            or set(np.unique(y_train)) != {0., 1.} or l2 <= 0):
        raise ValueError('Require finite aligned training ranks, binary labels, and OOF logits')
    mean, scale = float(rank_oof.mean()), max(float(rank_oof.std()), 1e-8)
    x = (rank_oof - mean) / scale
    result = minimize(objective, np.zeros(2), args=(x, y_train, anchor_oof, l2),
                      method='BFGS', jac=True, options=dict(gtol=1e-10, maxiter=200))
    value, gradient = objective(result.x, x, y_train, anchor_oof, l2)
    if not np.isfinite(result.x).all() or abs(gradient).max() > 1e-7:
        raise ValueError('Residual calibration did not reach the registered gradient tolerance')
    return dict(mean=mean, scale=scale, beta=float(result.x[0]), intercept=float(result.x[1]),
                l2=l2, gradient_max_abs=float(abs(gradient).max()), objective=value,
                iterations=int(result.nit), training_rows=len(y_train))


def predict(rank, calibration, bound=2.):
    rank = np.asarray(rank, dtype=np.float64)
    if not np.isfinite(rank).all() or bound <= 0:
        raise ValueError('Require finite ranks and a positive bound')
    value = calibration['beta'] * (rank - calibration['mean']) / calibration['scale']
    return np.clip(value + calibration['intercept'], -bound, bound)
