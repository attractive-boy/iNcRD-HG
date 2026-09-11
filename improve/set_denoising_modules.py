"""Drug-conditioned positive-set denoising and its 2 x 2 controls.

This is a small task-specific masked-set prototype, not a reproduction of
RADD or KREPE. All RNA candidates remain present. A sampled zero is an
unobserved benchmark label, never a confirmed negative support token.
"""
import math

import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
from torch.nn import functional as F


ARMS = {
    'pointwise': dict(interaction=False, denoising=False),
    'set_plain': dict(interaction=True, denoising=False),
    'point_denoise': dict(interaction=False, denoising=True),
    'set_denoise': dict(interaction=True, denoising=True),
}


def fold_assignments(pairs, labels, folds, seed):
    if pairs.shape != (2, len(labels)):
        raise ValueError('Expected pairs of shape (2, n)')
    if len(set(map(tuple, pairs.T))) != len(labels):
        raise ValueError('Duplicate pair; split unique pairs before cross-fitting')
    ids = np.empty(len(labels), dtype=np.int64)
    for f, (_, query) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(pairs.T, labels)):
        ids[query] = f
    return ids


def support_and_queries(pairs, labels, fold_ids, fold, nr, nd):
    """Positive-only support and supervision masks in drug-major order."""
    support = np.zeros((nd, nr), dtype=bool)
    query = np.zeros_like(support)
    target = np.zeros((nd, nr), dtype=np.float32)
    fit = np.arange(len(labels)) if fold is None else np.flatnonzero(fold_ids != fold)
    pos = fit[labels[fit] == 1]
    support[pairs[1, pos], pairs[0, pos]] = True
    if fold is not None:
        idx = np.flatnonzero(fold_ids == fold)
        query[pairs[1, idx], pairs[0, idx]] = True
        target[pairs[1, idx], pairs[0, idx]] = labels[idx]
    if np.any(support & query):
        raise AssertionError('A training query leaked into support')
    return support, query, target, fit


def unknown_grid_features(encoder, support_positive):
    """Return features for EVERY unknown pair, plus explicit support tokens.

    The established encoder deliberately rejects an observed positive as its
    own query. Known support positions therefore have zero structural features
    and are represented by their positive flag and RNA/drug attributes. They
    are never supervised or evaluated as unknown queries in this scenario.
    """
    support = np.zeros((encoder.nd, encoder.nr), dtype=bool)
    support[support_positive[1], support_positive[0]] = True
    d, r = np.nonzero(~support)
    pairs = np.stack([r, d]).astype(np.int64)
    frame = encoder.transform(pairs, support_positive)
    return pairs, frame.values, list(frame.columns), support


def corrupt_positive_set(support, query, target, time, uniform):
    """Mask each query positive with probability t; negatives stay unknown.

    The base support/teacher excludes the WHOLE query fold. Revealed query
    positives are used only as additional denoising context and receive no
    prediction loss. Hidden positive loss is weighted by 1/t, avoiding a
    changed expected class balance relative to the t=1 control.
    """
    if bool(torch.any(support & query)):
        raise ValueError('Support and query overlap')
    if bool(torch.any((time <= 0) | (time > 1))):
        raise ValueError('Mask probability must be in (0, 1]')
    positive = query & (target > .5)
    revealed = positive & (uniform >= time[:, None])
    observed = support | revealed
    loss_mask = query & ~revealed
    importance = torch.where(target > .5, 1 / time[:, None], torch.ones_like(target))
    weight = loss_mask.to(target.dtype) * importance
    return observed, weight, revealed


def drug_balanced_loss(logits, base_logits, target, query, weight, residual_penalty):
    """Same empirical benchmark-label objective for all four arms."""
    denom = query.sum(1).clamp_min(1).to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    residual = (logits - base_logits).square()
    return (((bce + residual_penalty * residual) * weight).sum(1) / denom).mean()


class CandidateBlock(nn.Module):
    """O(n * slots) shared set mixing, or a parameter-matched local control.

    Both paths use all projection weights. The local control normalizes its
    key across slots within each token and cannot read other candidate rows.
    No learned RNA IDs or positional embeddings are introduced.
    """
    def __init__(self, width, slots, dropout):
        super().__init__()
        self.key = nn.Linear(width, slots, bias=False)
        self.query = nn.Linear(width, slots)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * width, width))
        self.dropout = nn.Dropout(dropout)
        self.slots = slots

    def forward(self, h, interaction):
        z = self.norm1(h)
        key, query, value = self.key(z), torch.softmax(self.query(z), dim=-1), self.value(z)
        if interaction:
            pool = torch.softmax(key, dim=1).transpose(1, 2) @ value
            message = query @ pool
        else:
            gate = self.slots * (torch.softmax(key, dim=-1) * query).sum(-1, keepdim=True)
            message = value * gate
        h = h + self.dropout(self.output(message))
        return h + self.dropout(self.ffn(self.norm2(h)))


class SetDenoiser(nn.Module):
    def __init__(self, feature_dim, rna_dim, drug_dim, width=64, depth=2,
                 slots=8, dropout=.1, residual_cap=2.):
        super().__init__()
        self.feature = nn.Linear(feature_dim + 1, width)
        self.rna = nn.Linear(rna_dim, width, bias=False)
        self.drug = nn.Linear(drug_dim, width, bias=False)
        self.observed = nn.Embedding(2, width)
        self.time = nn.Linear(4, width, bias=False)
        self.norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([CandidateBlock(width, slots, dropout) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)
        self.residual_cap = residual_cap
        # The untouched candidate is an exact, reviewable TabM baseline.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features, base_logits, rna, drug, observed, time, interaction=True):
        if features.shape[:2] != base_logits.shape or observed.shape != base_logits.shape:
            raise ValueError('Candidate dimensions do not match')
        # All visible positives, including those revealed during corruption,
        # have the same context-token convention as cached support positives.
        features = features.masked_fill(observed[..., None], 0.)
        prior = base_logits.masked_fill(observed, 0.).clamp(-12, 12)[..., None] / 6.
        time_features = torch.stack([time, 1 - time, torch.sin(math.pi * time), torch.cos(math.pi * time)], -1)
        h = (self.feature(torch.cat([features, prior], -1)) + self.rna(rna)[None]
             + self.drug(drug)[:, None] + self.observed(observed.long())
             + self.time(time_features)[:, None])
        h = self.norm(h)
        for block in self.blocks:
            h = block(h, interaction)
        residual = self.residual_cap * torch.tanh(self.head(self.out_norm(h)).squeeze(-1))
        return base_logits + residual


def ranking_metrics(y, score, pairs):
    result = dict(auc=float(roc_auc_score(y, score)), ap=float(average_precision_score(y, score)))
    rows = []
    for d in np.unique(pairs[1]):
        mask = pairs[1] == d
        if len(np.unique(y[mask])) < 2:
            continue
        yy, ss = y[mask], score[mask]
        order = np.argsort(-ss, kind='stable')
        k = min(10, len(yy))
        rows.append(dict(drug=int(d), ap=float(average_precision_score(yy, ss)),
            auc=float(roc_auc_score(yy, ss)), ndcg10=float(ndcg_score(yy[None], ss[None], k=k)),
            recall10=float(yy[order[:k]].sum() / yy.sum()),
            positives=int(yy.sum()), candidates=int(len(yy))))
    if not rows:
        raise ValueError('No eligible drug for the predeclared selection criterion')
    for name in ('ap', 'auc', 'ndcg10', 'recall10'):
        result['macro_drug_' + name] = float(np.mean([r[name] for r in rows]))
    result['macro_drugs'] = len(rows)
    result['criterion'] = .5 * result['macro_drug_ap'] + .25 * (result['auc'] + result['ap'])
    return result, rows


def full_candidate_recall(y, pairs, grid, support, k=20):
    """Held-out positive recall among ALL unknown RNAs, not sampled negatives.

    Unlabeled candidates are competitors, not confirmed false positives.
    This measures recovery of the known held-out subset only.
    """
    values = []
    for d in np.unique(pairs[1, y == 1]):
        positives = pairs[0, (pairs[1] == d) & (y == 1)]
        if np.any(support[d, positives]):
            raise ValueError('Held-out positive is in full-candidate support')
        scores = grid[d].copy()
        scores[support[d]] = -np.inf
        order = np.argsort(-scores, kind='stable')[:min(k, int((~support[d]).sum()))]
        values.append(float(np.isin(positives, order).mean()))
    return {'full_candidate_macro_recall'+str(k): float(np.mean(values)),
            'full_candidate_recall_drugs': len(values)}


def choose_checkpoint(trace, window=3):
    """Select a single checkpoint; step zero competes with trained candidates."""
    zero = trace[0]
    if zero['step'] != 0:
        raise ValueError('Step zero is required')
    candidates = [(zero['criterion'], 0)]
    trained = trace[1:]
    for i in range(window - 1, len(trained)):
        value = float(np.mean([r['criterion'] for r in trained[i-window+1:i+1]]))
        candidates.append((value, trained[i]['step']))
    value, step = sorted(candidates, key=lambda x: (-x[0], x[1]))[0]
    return step, value
