"""Episodic graph learning through a differentiable, support-fitted ridge head.

Inspired by Mochi (arXiv:2604.22031), without its pretrained weights or claims
of universal graph transfer. Every episode separates graph facts, conditioning
labels, and optimization queries. The baseline reference is an external offset.
"""
import itertools

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import overlap_block_models as graph_lib
from improve.sharpness_graph_v1 import identity_task, IdentityResidual

KINDS = ('full', 'stop_support_gradient')


def parameters():
    return dict(width=64, heads=4, depth=2, dropout=.1, folds=5,
        ridge_lambda=10., residual_bound=2., score_scale=2.,
        lnc_policy='identity_strong_reference', conditioning='one_fold',
        graph_support='three_other_folds', query='remaining_fold',
        ridge_bias_penalized=True, solve_precision='float64')


def ridge_scores(support, labels, query, ridge_lambda, stop_support_gradient=False):
    """Primal solve exactly equivalent to the published dual ridge objective.

    The bias is regularized, as in Mochi Eq. (1). No inverse is formed.
    A stopped support gradient is only an ablation, not another proper objective.
    """
    if (support.ndim != 2 or query.ndim != 2 or support.shape[1] != query.shape[1]
            or labels.shape != (len(support),) or ridge_lambda <= 0):
        raise ValueError('Invalid conditional ridge problem')
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError('Only registered binary conditioning labels are accepted')
    s = support.detach() if stop_support_gradient else support
    s, q = s.double(), query.double()
    s = torch.cat((s, torch.ones_like(s[:, :1])), dim=1)
    q = torch.cat((q, torch.ones_like(q[:, :1])), dim=1)
    y = F.one_hot(labels.long(), num_classes=2).double()
    gram = s.T @ s + ridge_lambda * torch.eye(s.shape[1], device=s.device, dtype=s.dtype)
    weights = torch.linalg.solve(gram, s.T @ y)
    return q @ weights


class EpisodicKernelGraph(nn.Module):
    def __init__(self, dimensions, relations, p, kind):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unregistered episodic arm')
        self.kind, self.p = kind, dict(p)
        w = p['width']
        self.project = nn.ModuleList([nn.Linear(d + relations, w) for d in dimensions])
        self.norm = nn.LayerNorm(w)
        self.layers = nn.ModuleList([graph_lib.SparseTypedAttention(w, p['heads'],
            2 * relations + 2, p['dropout']) for _ in range(p['depth'])])
        self.pair_encoder = nn.Sequential(nn.Linear(4 * w, w), nn.GELU(), nn.Linear(w, w))
        self.log_temperature = nn.Parameter(torch.tensor(0.))
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()

    def encode(self, xs, graph):
        edges, relation, counts = graph
        parts, start = [], 0
        for x, project in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            parts.append(project(torch.cat((x / rms, counts[start:start + len(x)]), dim=-1)))
            start += len(x)
        h = F.gelu(self.norm(torch.cat(parts)))
        for layer in self.layers:
            h = layer(h, edges, relation)
        return h

    def pair_features(self, h, pairs):
        left, right = h[pairs[0]], h[pairs[1]]
        z = self.pair_encoder(torch.cat((left, right, left * right, (left - right).abs()), dim=-1))
        return F.normalize(z, dim=-1) * self.log_temperature.clamp(-3., 3.).exp()

    def forward(self, xs, episode, pairs):
        h = self.encode(xs, episode['graph'])
        zs = self.pair_features(h, episode['support_pairs'])
        zq = self.pair_features(h, pairs)
        scores = ridge_scores(zs, episode['support_y'], zq, self.p['ridge_lambda'],
                              self.kind == 'stop_support_gradient')
        contrast = scores[:, 1] - scores[:, 0]
        return self.p['residual_bound'] * torch.tanh(self.p['score_scale'] * contrast)


def setup(a, p, kind, device):
    if kind not in KINDS or p['folds'] != 5:
        raise ValueError('Unregistered episodic configuration')
    groups = [np.flatnonzero(a['folds'] == f) for f in range(p['folds'])]
    if any(not len(ids) for ids in groups):
        raise ValueError('Every training fold must be populated')
    if identity_task(a, p):
        folds = [torch.as_tensor(ids, device=device) for ids in groups]
        return IdentityResidual().to(device), [], [None] * p['folds'], folds
    n, relations = int(a['node_counts'].sum()), int(a['relation'].max()) + 1
    raw_graphs = {}
    for left, right in itertools.combinations(range(p['folds']), 2):
        deleted = np.concatenate((groups[left], groups[right]))
        edges, relation = graph_lib.masked_physical(a, deleted)
        raw_graphs[left, right] = graph_lib.graph_inputs(edges, relation,
            np.empty((2, 0), dtype=np.int64), n, relations, device)
    episodes, folds = [], []
    for query_fold in range(p['folds']):
        for support_fold in range(p['folds']):
            if query_fold == support_fold:
                continue
            support_ids, query_ids = groups[support_fold], groups[query_fold]
            if np.intersect1d(support_ids, query_ids).size:
                raise ValueError('Conditioning and optimization queries overlap')
            episodes.append(dict(graph=raw_graphs[tuple(sorted((query_fold, support_fold)))],
                support_pairs=torch.as_tensor(a['pairs_train'][:, support_ids] + a['pair_offset'][:, None], device=device),
                support_y=torch.as_tensor(a['y_train'][support_ids], device=device),
                query_ids=query_ids.copy(), support_ids=support_ids.copy(),
                query_fold=query_fold, support_fold=support_fold))
            folds.append(torch.as_tensor(query_ids, device=device))
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = EpisodicKernelGraph([x.shape[1] for x in xs], relations, p, kind).to(device)
    return model, xs, episodes, folds
