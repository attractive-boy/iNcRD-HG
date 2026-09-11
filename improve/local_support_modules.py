"""Bounded pair-local case corrections around the original GNN predictor.

The original graph encoder and concat decoder remain the prediction prior.
Queries retrieve fitting cases using BOTH RNA and drug input features. A
regularized local Newton step uses support prediction errors, not raw labels
as a replacement classifier. Query cases must be absent from graph and memory.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from improve.support_operator_modules import exclude_support_pairs


def local_newton_step(features, labels, probabilities, weights, ridge):
    """One regularized logistic Newton step in a local linear feature basis.

    Inputs have shapes [queries, neighbors, rank] and [queries, neighbors].
    The caller detaches support prior probabilities: this is an error-fitting
    adaptation step, not a gradient path for manipulating support confidence.
    """
    if features.ndim != 3 or labels.shape != features.shape[:2]:
        raise ValueError('Invalid local support shapes')
    if labels.shape != probabilities.shape or weights.shape != labels.shape or ridge <= 0:
        raise ValueError('Invalid probabilities, weights, or regularization')
    precision = weights * probabilities * (1 - probabilities)
    transpose = features.transpose(1, 2)
    gram = torch.bmm(transpose * precision[:, None, :], features)
    rhs = torch.bmm(transpose, (weights * (labels - probabilities))[:, :, None])
    eye = torch.eye(features.shape[-1], dtype=features.dtype, device=features.device)
    return torch.linalg.solve(gram + ridge * eye, rhs).squeeze(-1)


class LocalSupportAdapter(nn.Module):
    """Original node features select cases; the prior remains the full GNN.

    Fixed tanh clipping limits the change to each logit. The bound is a
    numerical trust region, not a guarantee of improved prediction quality.
    Allocated parameters are shared across controls, but active capacity is
    not equal when keys or the optional MLP are unused.
    """
    MODES = ('case_joint', 'case_drug', 'local_mlp')

    def __init__(self, rna_dim, drug_dim, rna_type, rank=16):
        super().__init__()
        self.rna_type, self.rank = rna_type, rank
        self.rna_key = nn.Sequential(nn.LayerNorm(rna_dim), nn.Linear(rna_dim, rank), nn.Tanh())
        self.drug_key = nn.Sequential(nn.LayerNorm(drug_dim), nn.Linear(drug_dim, rank), nn.Tanh())
        self.rna_basis = nn.Sequential(nn.LayerNorm(rna_dim), nn.Linear(rna_dim, 2 * rank))
        self.drug_basis = nn.Sequential(nn.LayerNorm(drug_dim), nn.Linear(drug_dim, 2 * rank))
        self.local_head = nn.Sequential(nn.Linear(rank + 1, rank), nn.LeakyReLU(.1), nn.Linear(rank, 1))
        nn.init.zeros_(self.local_head[-1].weight)
        nn.init.zeros_(self.local_head[-1].bias)

    def basis(self, rna, drug, pairs):
        rmain, rproduct = self.rna_basis(rna).chunk(2, dim=-1)
        dmain, dproduct = self.drug_basis(drug).chunk(2, dim=-1)
        r, d = pairs
        values = torch.tanh(rmain[r] + dmain[d] + rproduct[r] * dproduct[d]) / math.sqrt(self.rank)
        return torch.cat((values, torch.ones_like(values[:, :1])), dim=-1)

    def forward(self, x, queries, support_pairs, support_labels, support_prior,
                *, mode='case_joint', topk=32, ridge=.1, temperature=.2,
                max_correction=.25, ramp=1., intervention=None, return_details=False):
        if mode not in self.MODES or topk < 1 or ridge <= 0 or temperature <= 0 or max_correction <= 0:
            raise ValueError('Invalid adapter mode or hyperparameters')
        if not 0 <= ramp <= 1:
            raise ValueError('Ramp must lie in [0,1]')
        if intervention not in (None, 'no_correction', 'uniform_selected', 'drug_only'):
            raise ValueError(intervention)
        rna, drug = x[self.rna_type], x['drug']
        support, labels, keep = exclude_support_pairs(support_pairs, support_labels, queries, len(drug))
        prior = support_prior[keep].detach()
        if not bool(torch.isfinite(labels).all()) or not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError('Support labels must be binary dataset labels')
        # Canonical pair order makes top-k tie handling independent of caller
        # order. The runner validates unique fitting pairs before training.
        order = torch.argsort(support[0] * len(drug) + support[1])
        support, labels, prior = support[:, order], labels[order], prior[order]
        feature_pairs = torch.cat((queries, support), dim=1)
        features = self.basis(rna, drug, feature_pairs)
        query_features, support_features = features[:queries.shape[1]], features[queries.shape[1]:]
        correction = query_features.sum(-1) * 0
        details = dict(support_rows=support.shape[1], selected_pairs=None, raw_correction=correction)
        if intervention == 'no_correction' or ramp == 0:
            pass
        elif mode == 'local_mlp':
            raw = self.local_head(query_features).flatten()
            correction = ramp * max_correction * raw.tanh()
            details['raw_correction'] = raw
        elif support.shape[1]:
            rk = F.normalize(self.rna_key(rna), dim=-1, eps=1e-8)
            dk = F.normalize(self.drug_key(drug), dim=-1, eps=1e-8)
            similarities = dk[queries[1]] @ dk[support[1]].T
            if mode == 'case_joint' and intervention != 'drug_only':
                similarities = .5 * (similarities + rk[queries[0]] @ rk[support[0]].T)
            scores, ids = torch.topk(similarities, min(topk, support.shape[1]), dim=1)
            weights = (torch.full_like(scores, 1 / scores.shape[1]) if intervention == 'uniform_selected'
                       else torch.softmax(scores / temperature, dim=1))
            local_features = support_features[ids]
            delta = local_newton_step(local_features, labels[ids].to(local_features.dtype),
                                      prior.sigmoid()[ids], weights, ridge)
            raw = (query_features * delta).sum(-1)
            correction = ramp * max_correction * raw.tanh()
            details.update(selected_pairs=support[:, ids], raw_correction=raw,
                           weights=weights, delta=delta)
        if return_details:
            return correction, details
        return correction


class LocalSupportModel(nn.Module):
    """Full original predictor plus a case-conditioned local error correction."""
    def __init__(self, base, data, rna_type, config, mode='case_joint', rank=16):
        super().__init__()
        self.base, self.mode, self.config = base, mode, config
        self.adapter = None if mode == 'pca_query' else LocalSupportAdapter(
            data[rna_type].x.shape[1], data['drug'].x.shape[1], rna_type, rank)

    def forward(self, x, edges, queries, weights, support_pairs, support_labels,
                ramp=1., intervention=None, return_components=False):
        z = self.base.encoder(x, edges, weights)
        prior = self.base.decoder(z, queries)
        correction = torch.zeros_like(prior)
        if self.adapter is not None:
            # A support case may be a known edge in this context graph. These
            # are conditional fitting residuals, not out-of-fold calibration
            # residuals. Query labels are absent from BOTH graph and support.
            with torch.no_grad():
                support_prior = self.base.decoder({k: v.detach() for k, v in z.items()}, support_pairs)
            correction = self.adapter(x, queries, support_pairs, support_labels, support_prior,
                mode=self.mode, topk=self.config['topk'], ridge=self.config['ridge'],
                temperature=self.config['temperature'], max_correction=self.config['max_correction'],
                ramp=ramp, intervention=intervention)
        if return_components:
            return prior + correction, prior, correction
        return prior + correction
