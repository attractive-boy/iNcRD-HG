"""Training-support row/column likelihood offsets for graph residual studies.

This regularized Rasch model is a classical nuisance baseline, not a proposed
innovation. The graph mechanism must beat both this baseline and its matched
mechanism control. Fitting uses benchmark observations, not biological negatives.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
import torch
from torch import nn


def likelihood(theta, pairs, labels, shape, penalty):
    nr, nd = shape
    r, d = pairs
    score = theta[r] + theta[nr + d] + theta[-1]
    residual = expit(score) - labels
    value = np.logaddexp(0., score).sum() - labels @ score + .5 * penalty * (theta[:-1] @ theta[:-1])
    gradient = np.concatenate((np.bincount(r, weights=residual, minlength=nr),
                               np.bincount(d, weights=residual, minlength=nd), [residual.sum()]))
    gradient[:-1] += penalty * theta[:-1]
    return float(value), gradient


def fit_effects(pairs, labels, shape, penalty=1.):
    pairs = np.asarray(pairs, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)
    nr, nd = map(int, shape)
    if pairs.shape != (2, len(labels)) or len(labels) < 2 or not np.isin(labels, (0, 1)).all():
        raise ValueError('Expected binary observed fitting pairs')
    if not 0 < labels.mean() < 1 or penalty <= 0:
        raise ValueError('Both observed classes and positive regularization are required')
    if (pairs < 0).any() or (pairs[0] >= nr).any() or (pairs[1] >= nd).any():
        raise ValueError('Pair endpoint outside the registered dictionary')
    order = np.argsort(pairs[0] * nd + pairs[1], kind='stable')
    pairs, labels = pairs[:, order], labels[order]
    if len(np.unique(pairs[0] * nd + pairs[1])) != len(labels):
        raise ValueError('Duplicate observations are not allowed')
    initial = np.zeros(nr + nd + 1)
    initial[-1] = logit(labels.mean())
    fit = minimize(likelihood, initial, args=(pairs, labels, (nr, nd), penalty), jac=True,
                   method='L-BFGS-B', options=dict(maxiter=500, ftol=1e-13, gtol=1e-7, maxls=40))
    _, gradient = likelihood(fit.x, pairs, labels, (nr, nd), penalty)
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError('Support-only offset did not converge: ' + str(fit.message))
    return fit.x[:nr], fit.x[nr:nr + nd], float(fit.x[-1]), dict(
        optimizer_success=bool(fit.success), iterations=int(fit.nit), objective=float(fit.fun),
        maximum_gradient=float(abs(gradient).max()), penalty=float(penalty), observations=len(labels))


def context_offsets(a, folds=5, penalty=1.):
    rna_type = int(np.flatnonzero(np.r_[0, np.cumsum(a['node_counts'])][:-1] == a['pair_offset'][0])[0])
    shape = (int(a['node_counts'][rna_type]), int(a['nd']))
    nodes, intercepts, diagnostics = [], [], []
    for fold in range(folds):
        fitting = a['folds'] != fold
        r, d, c, detail = fit_effects(a['pairs_train'][:, fitting], a['y_train'][fitting], shape, penalty)
        node = np.zeros(int(a['node_counts'].sum()), dtype=np.float64)
        ro, do = a['pair_offset']
        node[ro:ro + shape[0]] = r
        node[do:do + shape[1]] = d
        nodes.append(node)
        intercepts.append(c)
        diagnostics.append(dict(fold=fold, **detail))
    return np.stack(nodes), np.asarray(intercepts), diagnostics


def offset_prediction(a, pairs):
    query = pairs + a['pair_offset'][:, None]
    values = a['offset_nodes'][:, query[0]] + a['offset_nodes'][:, query[1]] + a['offset_intercepts'][:, None]
    return logit(np.clip(expit(values).mean(0), 1e-7, 1 - 1e-7))


class OffsetGraph(nn.Module):
    def __init__(self, core, family):
        super().__init__()
        if family not in ('block', 'geometry', 'state_space'):
            raise ValueError('Unsupported graph residual family')
        self.core, self.family = core, family
        # The independently fitted support offset replaces these learned
        # marginal heads identically in the full model and every control.
        for head in (core.rna_degree, core.drug_degree):
            for value in head.parameters():
                nn.init.zeros_(value)
                value.requires_grad_(False)

    @property
    def mode(self):
        return self.core.mode

    def encode(self, xs, graph):
        original, offset, intercept = graph
        value = self.core.encode(xs, original)
        if self.family == 'block':
            value['counts'] = (value['counts'], offset, intercept)
            return value
        h, counts = value
        return h, (counts, offset, intercept)

    def sample(self, posterior, generator, temperature=1., discrete=False):
        return self.core.sample(posterior, generator, temperature, discrete)

    def decode(self, h, packed_counts, pairs):
        counts, offset, intercept = packed_counts
        r, d = pairs
        return self.core.decode(h, counts, pairs).double() + offset[r] + offset[d] + intercept

    def forward(self, xs, graph, pairs, generator=None, temperature=1.):
        if self.family == 'block':
            posterior = self.encode(xs, graph)
            h, kl = self.sample(posterior, generator, temperature)
            return self.decode(h, posterior['counts'], pairs), kl
        return self.decode(*self.encode(xs, graph), pairs)
