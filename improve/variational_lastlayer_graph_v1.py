"""Correlated variational Bayesian last layers on query-deleted graph features.

Independent binary adaptation of arXiv:2404.11599v1, Eq. (14). The two class
weight vectors have independent Gaussian posteriors; each vector can have a
dense or diagonal covariance. Feature weights are point estimates. Logit
noise is set to zero, a case explicitly permitted in the paper. MAP uses
the same mean prior but is not described as a zero-KL delta posterior.
"""
from functools import lru_cache
import math

import numpy as np
import torch
from torch import nn

from improve import sharpness_graph_v1 as ordinary
from improve import nested_anchor_inputs_v2 as nested
from improve import overlap_block_models as graph_lib

KINDS = ('full', 'diagonal', 'map')
identity_task = ordinary.identity_task


def parameters():
    return dict(width=64, heads=4, depth=2, dropout=.1, folds=5,
                lnc_policy='identity_strong_reference', prior_scale=1.,
                quadrature_order=64, quadrature_reference_order=128,
                quadrature_tolerance=3e-5, covariance_initialization='std=1/(width+1)',
                logit_noise_variance=0., kl_multiplier=1.)


class GaussianClassifier(nn.Module):
    def __init__(self, width, mode, prior_scale=1.):
        super().__init__()
        if mode not in KINDS or prior_scale <= 0:
            raise ValueError('Unknown posterior or invalid prior')
        self.mode, self.dimension = mode, width+1
        d = self.dimension
        self.prior_variance = 2.*prior_scale/d
        self.mean = nn.Parameter(torch.zeros(2, d))
        self.log_scale = nn.Parameter(torch.full((2, d), -math.log(d)), requires_grad=mode != 'map')
        rows, columns = torch.tril_indices(d, d, offset=-1)
        self.register_buffer('lower_rows', rows)
        self.register_buffer('lower_columns', columns)
        self.lower = nn.Parameter(torch.zeros(2, len(rows)), requires_grad=mode == 'full')

    def factor(self):
        scale = self.log_scale.double().exp()
        factor = torch.diag_embed(scale)
        if self.mode == 'full':
            offdiagonal = factor.new_zeros(factor.shape)
            offdiagonal[:, self.lower_rows, self.lower_columns] = self.lower.double()
            factor = factor + offdiagonal
        return factor

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.dimension-1:
            raise ValueError('Unexpected last-layer feature shape')
        phi = torch.cat((features.double(), features.new_ones((len(features), 1)).double()), -1)
        mean = phi @ self.mean.double().T
        if self.mode == 'map':
            return mean, torch.zeros_like(mean)
        if self.mode == 'diagonal':
            variance = phi.square() @ (2*self.log_scale.double()).exp().T
        else:
            transformed = torch.einsum('nd,kde->kne', phi, self.factor())
            variance = transformed.square().sum(-1).T
        return mean, variance

    def regularizer(self):
        mean_penalty = .5*self.mean.double().square().sum()/self.prior_variance
        if self.mode == 'map':
            return mean_penalty
        trace = self.factor().square().sum()
        logdet = 2*self.log_scale.double().sum()
        return mean_penalty + .5*(trace/self.prior_variance - logdet +
                                 2*self.dimension*math.log(self.prior_variance) - 2*self.dimension)


class VariationalGraph(ordinary.OrdinaryGraph):
    def __init__(self, dimensions, relations, p, mode):
        super().__init__(dimensions, relations, p)
        # Construct the same encoder first for every arm; replacing the final
        # scalar map does not consume random numbers in the Bayesian head.
        self.readout[-1] = nn.Identity()
        self.last = GaussianClassifier(p['width'], mode, p['prior_scale'])

    def forward(self, xs, graph, pairs):
        features = super().forward(xs, graph, pairs)
        return self.last(features)


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Undeclared variational classifier arm')
    if set(a) - (nested.TRAIN_KEYS | {'anchor_oof'}):
        raise ValueError('Graph setup accepts training arrays only')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        return ordinary.IdentityResidual().to(device), [], [None]*p['folds'], folds
    if p['logit_noise_variance'] != 0. or p['covariance_initialization'] != 'std=1/(width+1)':
        raise ValueError('Unregistered noise or covariance initialization')
    n, relations = int(a['node_counts'].sum()), int(a['relation'].max())+1
    graphs = []
    for f in range(p['folds']):
        edges, labels = graph_lib.masked_physical(a, np.flatnonzero(a['folds'] == f))
        graphs.append(graph_lib.graph_inputs(edges, labels, np.empty((2, 0), dtype=np.int64), n, relations, device))
    xs = [torch.as_tensor(a['x_'+str(i)], device=device) for i in range(len(a['node_counts']))]
    model = VariationalGraph([x.shape[1] for x in xs], relations, p, kind).to(device)
    return model, xs, graphs, folds


def objective(mean, variance, offset, labels, penalty, total_training_rows):
    if (mean.ndim != 2 or mean.shape[1] != 2 or variance.shape != mean.shape
            or offset.shape != mean.shape[:1] or labels.shape != offset.shape or total_training_rows < 1):
        raise ValueError('Invalid variational objective inputs')
    logits = mean.double() + torch.stack((torch.zeros_like(offset), offset), -1).double()
    target = logits.gather(1, labels.long()[:, None]).squeeze(1)
    nll_bound = (torch.logsumexp(logits + .5*variance.double(), -1)-target).mean()
    regularizer = penalty.double()/total_training_rows
    return nll_bound + regularizer, nll_bound, regularizer


@lru_cache(maxsize=4)
def quadrature(order):
    if order not in (32, 64, 128, 256):
        raise ValueError('Unsupported quadrature order')
    points, weights = np.polynomial.hermite.hermgauss(order)
    return points, np.log(weights)-.5*np.log(np.pi)


def logistic_normal_logit(mean, variance, order=64):
    """One-dimensional Gauss--Hermite approximation, evaluated in log space."""
    if mean.shape != variance.shape or (variance < 0).any():
        raise ValueError('Invalid predictive Gaussian')
    points, log_weights = quadrature(order)
    points = torch.as_tensor(points, device=mean.device, dtype=torch.float64)
    log_weights = torch.as_tensor(log_weights, device=mean.device, dtype=torch.float64)
    sample = mean.double()[:, None] + (2*variance.double()).sqrt()[:, None]*points
    log_positive = torch.logsumexp(torch.nn.functional.logsigmoid(sample)+log_weights, -1)
    log_negative = torch.logsumexp(torch.nn.functional.logsigmoid(-sample)+log_weights, -1)
    return torch.where(variance == 0, mean.double(), log_positive-log_negative)


def residual_prediction(mean, variance, offset, order=64):
    difference = mean[:, 1].double()-mean[:, 0].double()
    predictive_variance = variance.double().sum(-1)
    latent_mean = offset.double()+difference
    posterior_logit = logistic_normal_logit(latent_mean, predictive_variance, order)
    residual = torch.where(predictive_variance == 0, difference, posterior_logit-offset.double())
    return residual, difference, predictive_variance
