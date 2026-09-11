"""Prototype of a support-conditioned ncRNA--drug prediction rule.

Training association cases parameterize a differentiable regularized linear
predictor in a learned RNA space. This replaces the concat decoder; it is not
an extra residual score. Ridge/meta-learning and hypernetworks are prior art.

Only fitting cases may enter ``support_pairs/support_labels``. The module
additionally excludes ALL scored pairs, regardless of label. Callers must also
prevent query labels from entering the graph encoder. The accompanying probe
removes the entire target association relation from that encoder.

Dataset zeros remain sampled unobserved labels, not biological sensitivity.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def exclude_support_pairs(pairs, labels, queries, n_drug):
    """Remove every matching support row, including duplicate/zero-label rows."""
    if pairs.ndim != 2 or pairs.shape[0] != 2 or queries.ndim != 2 or queries.shape[0] != 2:
        raise ValueError('Pair arrays must have shape [2, N]')
    if labels.ndim != 1 or labels.numel() != pairs.shape[1] or n_drug < 1:
        raise ValueError('Invalid support labels or number of drugs')
    if pairs.dtype != torch.long or queries.dtype != torch.long:
        raise ValueError('Pair IDs must be int64')
    for ids in (pairs, queries):
        if ids.numel() and (bool((ids < 0).any()) or bool((ids[1] >= n_drug).any())):
            raise ValueError('Pair IDs out of range')
    forbidden = queries[0] * n_drug + queries[1]
    keep = ~torch.isin(pairs[0] * n_drug + pairs[1], forbidden)
    return pairs[:, keep], labels[keep], keep


def regularized_operator(gram, rhs, prior, ridge):
    """Solve (G + lambda I) w = b + lambda w0, without a matrix inverse.

    ``gram`` is a positive semidefinite evidence covariance and ``ridge`` is
    strictly positive. With G=b=0 the result is the prior. Directions in the
    nullspace of G retain their prior component when b lies in range(G).
    """
    if gram.ndim != 3 or gram.shape[-2:] != (prior.shape[-1], prior.shape[-1]):
        raise ValueError('Invalid Gram matrix shape')
    if rhs.shape != prior.shape or gram.shape[0] != prior.shape[0]:
        raise ValueError('Incompatible right hand side or prior')
    if ridge.shape != (len(prior),) or not bool((ridge > 0).all()):
        raise ValueError('A positive regularizer is required for every drug')
    eye = torch.eye(prior.shape[-1], device=prior.device, dtype=prior.dtype)
    system = gram + ridge[:, None, None] * eye
    target = rhs + ridge[:, None] * prior
    return torch.linalg.solve(system, target.unsqueeze(-1)).squeeze(-1)


class SupportOperatorDecoder(nn.Module):
    """Drug-conditioned prior adapted by softly retrieved association cases.

    Each observed source drug first contributes mean sufficient statistics,
    so a frequently observed drug does not automatically dominate retrieval.
    Softmax weights then mix these drug-specific statistics. Normalizing the
    total evidence mass means more cases do not mechanically mean a stronger
    update; a learned ridge head also sees effective support size.

    Computing per-source sufficient statistics avoids materializing the
    [target-drug, support-case, rank, rank] tensor. Cached embeddings can be
    reused when the explicit support set changes, provided the encoder graph
    does not depend on that target support set. This is not machine unlearning.
    """
    MODES = ('adaptive', 'uniform', 'prior')

    def __init__(self, hidden, rna_type, rank=16, initial_ridge=.1):
        super().__init__()
        if rank < 1 or initial_ridge <= 1e-3:
            raise ValueError('Positive rank and initial ridge > 1e-3 required')
        self.rna_type, self.rank = rna_type, rank
        self.rna_basis = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, rank), nn.Tanh())
        self.drug_prior = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, rank),
                                        nn.Tanh(), nn.Linear(rank, rank + 1))
        self.drug_key = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, rank, bias=False))
        self.ridge_head = nn.Linear(hidden + 1, 1)
        nn.init.zeros_(self.ridge_head.weight)
        nn.init.constant_(self.ridge_head.bias, math.log(math.expm1(initial_ridge - 1e-3)))
        self.raw_temperature = nn.Parameter(torch.tensor(math.log(math.expm1(.5 - .05))))
        self.raw_scale = nn.Parameter(torch.tensor(math.log(math.expm1(1.))))
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(self, z, query_pairs, support_pairs, support_labels, mode='adaptive',
                excluded_pairs=None, return_details=False):
        if mode not in self.MODES:
            raise ValueError(mode)
        drug = z['drug']
        excluded = query_pairs if excluded_pairs is None else torch.cat((query_pairs, excluded_pairs), dim=1)
        support, labels, keep = exclude_support_pairs(support_pairs, support_labels, excluded, len(drug))
        if not bool(torch.isfinite(labels).all()) or not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError('Support labels must be binary dataset labels')
        basis = self.rna_basis(z[self.rna_type]) / math.sqrt(self.rank)
        basis = torch.cat((basis, torch.ones_like(basis[:, :1])), dim=1)
        prior = self.drug_prior(drug)
        counts = torch.bincount(support[1], minlength=len(drug)).to(basis.dtype)
        present = torch.nonzero(counts > 0, as_tuple=False).flatten()
        kernel = basis.new_zeros((len(drug), len(present)))
        effective = basis.new_zeros(len(drug))
        ridge = F.softplus(self.ridge_head(torch.cat((drug, effective[:, None]), dim=1))).flatten() + 1e-3
        operator = prior
        if mode != 'prior' and support.shape[1]:
            x, target = basis[support[0]], 2 * labels.to(basis.dtype) - 1
            p = basis.shape[1]
            source_gram = basis.new_zeros((len(drug), p, p)).index_add(
                0, support[1], x[:, :, None] * x[:, None, :])
            source_rhs = basis.new_zeros((len(drug), p)).index_add(
                0, support[1], x * target[:, None])
            source_gram = source_gram[present] / counts[present, None, None]
            source_rhs = source_rhs[present] / counts[present, None]
            if mode == 'uniform':
                kernel = basis.new_full((len(drug), len(present)), 1 / len(present))
            else:
                keys = F.normalize(self.drug_key(drug), dim=-1, eps=1e-8)
                temperature = F.softplus(self.raw_temperature) + .05
                kernel = torch.softmax(keys @ keys[present].T / temperature, dim=-1)
            effective = 1 / (kernel.square() / counts[present]).sum(dim=1).clamp_min(1e-12)
            ridge_input = torch.cat((drug, torch.log1p(effective)[:, None]), dim=1)
            ridge = F.softplus(self.ridge_head(ridge_input)).flatten() + 1e-3
            gram = torch.einsum('ds,spq->dpq', kernel, source_gram)
            rhs = kernel @ source_rhs
            operator = regularized_operator(gram, rhs, prior, ridge)
        logits = (basis[query_pairs[0]] * operator[query_pairs[1]]).sum(dim=1)
        logits = F.softplus(self.raw_scale) * logits + self.offset
        if not return_details:
            return logits
        return logits, dict(prior=prior, operator=operator, kernel=kernel, ridge=ridge,
                            effective_support=effective, source_drugs=present,
                            support_pairs=support, support_keep=keep, source_counts=counts)
