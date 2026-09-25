"""Direct bipartite graph scorers: kernel posterior, graph factorization, denoising.

These models do not consume the 36/62/76-column pair-feature tables. They operate
on node kernels, the observed pair likelihood, or masked association matrices.
Candidate mechanisms are hypotheses; importing this module implies no success.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def normalize_kernel(k):
    k = np.asarray(k, dtype=np.float64)
    k = (k + k.T) / 2
    scale = np.sqrt(np.maximum(np.diag(k), 1e-12))
    k = k / scale[:, None] / scale[None, :]
    np.fill_diagonal(k, 1.)
    return k


def feature_kernel(x):
    """PSD spherical RBF, including an identity for featureless nodes."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    unit = x / np.maximum(norms, 1e-12)
    k = np.exp((unit @ unit.T - 1.) / .5)
    k[np.diag_indices_from(k)] = 1.
    return normalize_kernel(k)


def spectral_features(k, rank):
    values, vectors = np.linalg.eigh(normalize_kernel(k))
    order = np.argsort(values)[::-1][:min(rank, len(values))]
    # Eigenvector signs are fixed so checkpoints are reproducible across LAPACKs.
    u = vectors[:, order]
    pivot = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[pivot, np.arange(u.shape[1])])
    u *= np.where(signs == 0, 1, signs)
    return (u * np.sqrt(np.maximum(values[order], 0))[None, :]).astype(np.float32)


def graph_operator(k, neighbors=20):
    """Symmetric normalized nonnegative kNN operator, diagonal removed."""
    k = np.asarray(k, dtype=np.float64).copy()
    np.fill_diagonal(k, 0.)
    m = min(neighbors, max(0, len(k) - 1))
    if m:
        ix = np.argsort(k, axis=1, kind='stable')[:, -m:]
        a = np.zeros_like(k)
        np.put_along_axis(a, ix, np.take_along_axis(k, ix, axis=1), axis=1)
        a = np.maximum(a, a.T)
    else:
        a = np.zeros_like(k)
    degree = np.maximum(a.sum(1), 1e-12)
    return (a / np.sqrt(degree[:, None] * degree[None, :])).astype(np.float32)


def masked_contexts(pairs, labels, shape, folds):
    """A fold query has neither a positive nor an observed-zero context entry."""
    pairs, labels, folds = np.asarray(pairs), np.asarray(labels), np.asarray(folds)
    result = []
    for f in np.unique(folds):
        support = folds != f
        positive = np.zeros(shape, dtype=np.float32)
        zero = np.zeros(shape, dtype=np.float32)
        p = pairs[:, support & (labels == 1)]
        n = pairs[:, support & (labels == 0)]
        positive[p[0], p[1]] = 1.
        zero[n[0], n[1]] = 1.
        q = pairs[:, ~support]
        if np.any(positive[q[0], q[1]]) or np.any(zero[q[0], q[1]]):
            raise ValueError('Query outcome present in denoising context')
        result.append((int(f), positive, zero))
    return result


class KernelLogistic(nn.Module):
    """MAP logistic function with additive identities and a product RKHS prior.

    The low-rank product is a computational approximation to a separable kernel,
    distinct from concatenating endpoint features into a tabular classifier.
    """
    def __init__(self, ur, ud, prevalence, interaction=True):
        super().__init__()
        self.register_buffer('ur', torch.as_tensor(ur))
        self.register_buffer('ud', torch.as_tensor(ud))
        self.row = nn.Parameter(torch.zeros(len(ur)))
        self.col = nn.Parameter(torch.zeros(len(ud)))
        self.bias = nn.Parameter(torch.tensor(float(math.log(prevalence/(1-prevalence)))))
        self.w = nn.Parameter(torch.zeros(ur.shape[1], ud.shape[1]), requires_grad=interaction)

    def full(self):
        return self.bias + self.row[:, None] + self.col[None, :] + self.ur @ self.w @ self.ud.T

    def forward(self, pairs):
        scores = self.full()
        return scores[pairs[0], pairs[1]]

    def penalty(self, node_reg, interaction_reg):
        return .5 * (node_reg * (self.row.square().sum() + self.col.square().sum()) +
                     interaction_reg * self.w.square().sum())


class GraphFactorization(nn.Module):
    """Degree effects plus a low-rank log-odds surface with graph Gaussian prior.

    Both observed classes enter the likelihood. Unlisted pairs incur no negative
    loss. The graph prior acts on factors, not by treating unknowns as negatives.
    """
    def __init__(self, sr, sd, prevalence, rank=16):
        super().__init__()
        self.register_buffer('sr', torch.as_tensor(sr))
        self.register_buffer('sd', torch.as_tensor(sd))
        self.row = nn.Parameter(torch.zeros(len(sr)))
        self.col = nn.Parameter(torch.zeros(len(sd)))
        self.bias = nn.Parameter(torch.tensor(float(math.log(prevalence/(1-prevalence)))))
        self.u = nn.Parameter(torch.randn(len(sr), rank) * .02)
        self.v = nn.Parameter(torch.randn(len(sd), rank) * .02)

    def full(self):
        return self.bias + self.row[:, None] + self.col[None, :] + self.u @ self.v.T

    def forward(self, pairs):
        return self.bias + self.row[pairs[0]] + self.col[pairs[1]] + (self.u[pairs[0]]*self.v[pairs[1]]).sum(-1)

    def penalty(self, node_reg, interaction_reg, graph_reg):
        ridge = .5 * (node_reg * (self.row.square().sum() + self.col.square().sum()) +
                      interaction_reg * (self.u.square().sum() + self.v.square().sum()))
        smooth = .5 * graph_reg * (
            (self.u * (self.u - self.sr @ self.u)).sum() +
            (self.v * (self.v - self.sd @ self.v)).sum())
        return ridge + smooth


class AssociationDenoiser(nn.Module):
    """A graph-conditioned denoising network on complete bipartite row signals.

    Training query folds are absent in both input channels before graph
    smoothing. Continuous heat-time conditioning changes the smoothing scale.
    This is a masked heat-denoising adaptation, not a reproduction of GiffCF's
    complete generative sampler and not the older candidate-set score denoiser.
    """
    def __init__(self, ur, ud, sr, sd, prevalence, width=64, graph=True, zeros=True):
        super().__init__()
        self.register_buffer('ur', torch.as_tensor(ur))
        self.register_buffer('ud', torch.as_tensor(ud))
        self.register_buffer('sr', torch.as_tensor(sr))
        self.register_buffer('sd', torch.as_tensor(sd))
        self.use_graph, self.use_zeros = graph, zeros
        nr, nd = len(ur), len(ud)
        self.row_encoder = nn.Sequential(nn.Linear(nd*2+ur.shape[1]+4, width), nn.GELU(),
                                         nn.Dropout(.1), nn.Linear(width, width), nn.GELU())
        self.col_encoder = nn.Sequential(nn.Linear(nr*2+ud.shape[1]+4, width), nn.GELU(),
                                         nn.Dropout(.1), nn.Linear(width, width), nn.GELU())
        self.row_decoder = nn.Linear(width, nd)
        self.col_decoder = nn.Linear(width, nr)
        self.register_buffer('initial_logit', torch.tensor(float(math.log(prevalence/(1-prevalence)))))
        nn.init.normal_(self.row_decoder.weight, std=.01)
        nn.init.normal_(self.col_decoder.weight, std=.01)
        nn.init.zeros_(self.row_decoder.bias)
        nn.init.zeros_(self.col_decoder.bias)

    def full(self, positive, zero, time=0., corrupt=False):
        a, n = positive, zero if self.use_zeros else torch.zeros_like(zero)
        if corrupt:
            keep = (torch.rand_like(a) >= .1).to(a.dtype)
            a, n = a * keep, n * keep
        if self.use_graph:
            # Truncated nonnegative heat kernel e^{-t(I-S)} to second order.
            # t=0.5 is used at inference, and is inside the training time range.
            t = float(time)
            a = math.exp(-t) * (a + .5*t*(self.sr@a + a@self.sd) +
                               .125*t*t*(self.sr@(self.sr@a) + 2*(self.sr@a)@self.sd + (a@self.sd)@self.sd))
            n = math.exp(-t) * (n + .5*t*(self.sr@n + n@self.sd))
        row_stats = torch.stack([torch.log1p(positive.sum(1)), torch.log1p(zero.sum(1)),
                                 torch.full_like(positive.sum(1), float(time)),
                                 torch.ones_like(positive.sum(1))], 1)
        col_stats = torch.stack([torch.log1p(positive.sum(0)), torch.log1p(zero.sum(0)),
                                 torch.full_like(positive.sum(0), float(time)),
                                 torch.ones_like(positive.sum(0))], 1)
        r = self.row_encoder(torch.cat([a, n, self.ur, row_stats], 1))
        d = self.col_encoder(torch.cat([a.T, n.T, self.ud, col_stats], 1))
        return self.initial_logit + .5 * (self.row_decoder(r) + self.col_decoder(d).T)


def configurations():
    """A small frozen grid. Control variants never count as extra mechanisms."""
    configs = []
    for reg in (1., 10., 100.):
        configs.append(dict(id='kernel_'+str(int(reg)), family='kernel', rank=64,
                            node_reg=1., interaction_reg=reg, lr=.03, epochs=500))
    configs.append(dict(id='additive_control', family='kernel', rank=64,
                        node_reg=1., interaction_reg=1., interaction=False, lr=.03, epochs=500))
    for reg in (1., 10.):
        for smooth in (0., 10.):
            configs.append(dict(id=f'factor_{int(reg)}_g{int(smooth)}', family='factor', rank=16,
                                node_reg=1., interaction_reg=reg, graph_reg=smooth, lr=.03, epochs=600))
    for use_graph, zeros in ((False, True), (True, True), (True, False)):
        configs.append(dict(id=f'denoise_g{int(use_graph)}_z{int(zeros)}', family='denoise', rank=32,
                            graph=use_graph, zeros=zeros, width=64, lr=.001, epochs=500,
                            weight_decay=.01))
    return configs
