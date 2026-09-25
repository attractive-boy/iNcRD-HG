"""Conditional relation attention for a frozen TabICLv2 final ICL layer.

The adapter has one set of weights and no task/type/identifier input. Entity
indices only look up evidence and test whether a query shares an endpoint with
a support example. Query labels are never an argument to a prediction method.
"""
from dataclasses import dataclass
from contextlib import nullcontext
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve.tabicl_reli_modules import aggregate_members, decode


RELATION_DIM = 5
QUERY_DIM = 8


@dataclass
class RelationEvidence:
    rna_kernel: np.ndarray
    drug_kernel: np.ndarray
    rna_present: np.ndarray
    drug_present: np.ndarray

    def __post_init__(self):
        for name in ('rna', 'drug'):
            kernel = np.asarray(getattr(self, name + '_kernel'), dtype=np.float32)
            present = np.asarray(getattr(self, name + '_present'), dtype=bool)
            if kernel.shape != (len(present), len(present)) or not np.isfinite(kernel).all():
                raise ValueError('Expected a finite square entity kernel')
            setattr(self, name + '_kernel', kernel)
            setattr(self, name + '_present', present)

    def features(self, queries, support):
        queries, support = np.asarray(queries), np.asarray(support)
        for pairs in (queries, support):
            if pairs.ndim != 2 or pairs.shape[0] != 2 or pairs.dtype.kind not in 'iu' or np.any(pairs < 0):
                raise ValueError('Expected nonnegative integer pair columns')
            if not pairs.shape[1]:
                raise ValueError('Empty pair sets are not supported')
            if pairs[0].max() >= len(self.rna_kernel) or pairs[1].max() >= len(self.drug_kernel):
                raise ValueError('Entity index outside evidence')
        if len(set(map(tuple, support.T))) != support.shape[1]:
            raise ValueError('Duplicate support pairs')
        if set(map(tuple, queries.T)) & set(map(tuple, support.T)):
            raise ValueError('Query/support target pairs must be disjoint')
        standardized, availability, spread = [], [], []
        for axis, name in enumerate(('rna', 'drug')):
            kernel, present = getattr(self, name + '_kernel'), getattr(self, name + '_present')
            raw = kernel[queries[axis, :, None], support[axis, None, :]]
            mask = present[queries[axis, :, None]] & present[support[axis, None, :]]
            count = mask.sum(-1, keepdims=True)
            mean = (raw * mask).sum(-1, keepdims=True) / np.maximum(count, 1)
            variance = (((raw - mean) * mask) ** 2).sum(-1, keepdims=True) / np.maximum(count, 1)
            z = np.clip((raw - mean) / np.maximum(np.sqrt(variance), 1e-6), -3., 3.)
            standardized.append(np.where(mask, z, 0.))
            availability.append(mask.mean(-1))
            spread.append(np.clip(np.sqrt(variance[:, 0]), 0., 1.))
        same_rna = queries[0, :, None] == support[0, None, :]
        same_drug = queries[1, :, None] == support[1, None, :]
        n_rna, n_drug = same_rna.sum(-1), same_drug.sum(-1)
        rna, drug = standardized
        # The interaction distinguishes evidence agreeing on both endpoints
        # from an additive score that can be large on just one endpoint.
        relation = np.stack((rna, drug, np.clip(rna * drug, -3., 3.),
                             same_rna.astype(np.float32), same_drug.astype(np.float32)), -1)
        query = np.stack((n_rna / (n_rna + 4.), n_drug / (n_drug + 16.),
                          *availability, *spread), -1)
        return relation.astype(np.float32), query.astype(np.float32)


class ConditionalRelationAdapter(nn.Module):
    """At most two attention-logit units, with coefficients conditional on evidence.

    Six evidence/coverage statistics and the native classifier's uncertainty
    describe a query. No RNA length, RNA type, task ID or entity embedding can
    enter this controller. Both output layers start at zero for native recovery.
    """

    def __init__(self, conditional=True, width=16):
        super().__init__()
        self.conditional = bool(conditional)
        self.width = int(width)
        self.global_coefficients = nn.Parameter(torch.zeros(RELATION_DIM))
        self.controller = nn.Sequential(nn.Linear(QUERY_DIM, width), nn.Tanh(),
                                        nn.Linear(width, RELATION_DIM))
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)

    def forward(self, relation, evidence, native_logits, strength=1.):
        if relation.ndim != 3 or relation.shape[-1] != RELATION_DIM:
            raise ValueError('Expected query/support/relation features')
        if evidence.shape != (relation.shape[0], QUERY_DIM - 2):
            raise ValueError('Query evidence dimensions differ')
        if native_logits.ndim != 3 or native_logits.shape[1:] != (relation.shape[0], 2):
            raise ValueError('Native query logits differ')
        if not np.isfinite(strength) or not 0. <= strength <= 1.:
            raise ValueError('Strength must lie in [0, 1]')
        # Invariant to the official member's binary class permutation.
        margin = (native_logits[..., 1] - native_logits[..., 0]).abs().detach()
        uncertainty = 4. * torch.sigmoid(margin) * torch.sigmoid(-margin)
        inputs = torch.cat((evidence[None].expand(len(margin), -1, -1),
                            uncertainty[..., None], (margin / (1. + margin))[..., None]), -1)
        coefficients = self.global_coefficients[None, None]
        if self.conditional:
            coefficients = coefficients + self.controller(inputs)
        coefficients = torch.tanh(coefficients)
        bias = torch.einsum('bqf,qsf->bqs', coefficients.expand(len(margin), len(evidence), -1), relation)
        # A bounded differentiable change within attention, not a second output
        # classifier. Low confidence alone is not claimed to imply correctness.
        bias = 2. * torch.tanh(bias / (2. * RELATION_DIM ** .5))
        return float(strength) * uncertainty[..., None] * bias

    def config(self):
        return dict(conditional=self.conditional, width=self.width)


def prefix_cache(predictor, rows, labels):
    """Cache final-layer inputs while all support states remain query independent."""
    if predictor.training:
        raise ValueError('Frozen predictor must be in evaluation mode')
    with torch.no_grad():
        hidden = rows.clone()
        train_size = labels.shape[1]
        if not 0 < train_size < rows.shape[1]:
            raise ValueError('Expected support and query rows')
        hidden[:, :train_size] += predictor.y_encoder(labels.float())
        encoder = predictor.tf_icl
        for block in encoder.blocks[:-1]:
            hidden = block(q=hidden, train_size=train_size, rope=encoder.rope)
        native, key, value = encoder.blocks[-1](q=hidden, train_size=train_size,
                                                rope=encoder.rope, need_kv=True)
        return dict(query=hidden[:, train_size:].detach(), key=key.detach(), value=value.detach(),
                    native_logits=decode(predictor, native[:, train_size:], True, .9).detach())


def adapted_logits(predictor, cache, relation, evidence, adapter, strength=1.):
    from tabicl._model.kv_cache import KVCacheEntry
    bias = adapter(relation, evidence, cache['native_logits'], strength)
    block = predictor.tf_icl.blocks[-1]
    # The official attention layer requires the full head dimension for a
    # batched mask; expand is a view, not one copy per attention head.
    mask = bias[:, None].expand(-1, block.attn.num_heads, -1, -1)
    # PyTorch 2.5.1's memory-efficient CUDA backward rejects some short query
    # lengths with "LSE is not correctly aligned (strideH)". Only adapter
    # training uses the supported mathematical SDPA backend; evaluation keeps
    # the official inference kernels. No installed package is modified.
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backend = sdpa_kernel(SDPBackend.MATH) if torch.is_grad_enabled() and mask.is_cuda else nullcontext()
    with backend:
        changed = block(q=cache['query'], cached_kv=KVCacheEntry(key=cache['key'], value=cache['value']),
                        rope=predictor.tf_icl.rope, attn_mask=mask)
    return decode(predictor, changed, True, .9)


def load_adapter(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    adapter = ConditionalRelationAdapter(**state['config'])
    adapter.load_state_dict(state['state_dict'])
    return adapter.to(device).eval(), state


def predict_variants(clf, query_features, relation, evidence, variants, check_native=True):
    """Run actual TabICLv2 with several frozen shared adapter states.

    variants: {name: (adapter, strength)}. The returned ``zero`` is the official
    baseline, evaluated on precisely the same queries and inference members.
    """
    if 'zero' in variants or getattr(clf, 'model_kv_cache_', None) is not None:
        raise ValueError('Reserved variant name or unsupported existing KV cache')
    predictor, device = clf.model_.icl_predictor, torch.device(clf.device_)
    native_forward = predictor.forward
    relation = torch.as_tensor(relation, dtype=torch.float32, device=device)
    evidence = torch.as_tensor(evidence, dtype=torch.float32, device=device)
    captured = {name: [] for name in ('zero', *variants)}
    checks = []

    @torch.no_grad()
    def forward(this, R, y_train, return_logits=True, softmax_temperature=.9, mgr_config=None):
        if not return_logits:
            raise ValueError('This experiment fixes official average_logits=True')
        this.to(device)
        R, y_train = R.to(device), y_train.to(device)
        native = native_forward(R=R.clone(), y_train=y_train, return_logits=True,
                                softmax_temperature=softmax_temperature, mgr_config=mgr_config)
        cache = prefix_cache(this, R, y_train)
        error = float((cache['native_logits'] - native.to(device)).abs().max().cpu())
        if check_native:
            torch.testing.assert_close(cache['native_logits'], native.to(device), atol=1e-5, rtol=1e-5)
        checks.append(error)
        captured['zero'].append(native.float().cpu().numpy())
        for name, (adapter, strength) in variants.items():
            adapter.to(device).eval()
            value = adapted_logits(this, cache, relation, evidence, adapter, strength)
            captured[name].append(value.float().cpu().numpy())
        return native

    predictor.forward = types.MethodType(forward, predictor)
    try:
        official = np.asarray(clf.predict_proba(query_features))
    finally:
        predictor.forward = native_forward
    probabilities = {name: aggregate_members(clf, np.concatenate(parts)) for name, parts in captured.items()}
    np.testing.assert_allclose(probabilities['zero'], official, atol=1e-7, rtol=0)
    index = int(np.flatnonzero(np.asarray(clf.classes_) == 1)[0])
    return {name: p[:, index].astype(np.float64) for name, p in probabilities.items()}, dict(
        native_icl_max_abs=max(checks), forward_calls=len(checks),
        members=len(np.concatenate(captured['zero'])),
        aggregation_max_abs=float(np.max(np.abs(probabilities['zero'] - official))))


def capture_training_episode(clf, features, relation, evidence, targets):
    """Capture support-conditioned training pseudo-queries, with native labels aligned."""
    device, predictor = torch.device(clf.device_), clf.model_.icl_predictor
    native_forward = predictor.forward
    parts = []

    @torch.no_grad()
    def forward(this, R, y_train, return_logits=True, softmax_temperature=.9, mgr_config=None):
        this.to(device)
        cache = prefix_cache(this, R.to(device), y_train.to(device))
        parts.append({key: value.cpu() for key, value in cache.items()})
        return cache['native_logits']

    predictor.forward = types.MethodType(forward, predictor)
    try:
        clf.predict_proba(features)
    finally:
        predictor.forward = native_forward
    cache = {key: torch.cat([part[key] for part in parts]) for key in parts[0]}
    permutations = [p for group in clf.ensemble_generator_.class_shuffles_.values() for p in group]
    if len(permutations) != len(cache['query']):
        raise ValueError('Member class permutation mismatch')
    cache.update(relation=torch.from_numpy(relation), evidence=torch.from_numpy(evidence),
                 target=torch.as_tensor(np.asarray(targets), dtype=torch.long),
                 permutations=torch.as_tensor(np.asarray(permutations), dtype=torch.long))
    return cache


def aligned_logits(logits, permutations):
    if permutations.shape != (len(logits), 2):
        raise ValueError('Expected one binary permutation per inference member')
    return torch.gather(logits, -1, permutations[:, None].expand(-1, logits.shape[1], -1))


def adaptation_loss(logits, native_logits, targets, permutations, trust=1.):
    logits = aligned_logits(logits, permutations)
    native = aligned_logits(native_logits, permutations).detach()
    targets = targets[None].expand(len(logits), -1)
    ce = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))
    teacher = torch.softmax(native, -1)
    kl = (teacher * (torch.log_softmax(native, -1) - torch.log_softmax(logits, -1))).sum(-1).mean()
    # Per-member ranking uses only this training episode's pseudo-query labels.
    margin = logits[..., 1] - logits[..., 0]
    positive, negative = targets[0] == 1, targets[0] == 0
    if positive.any() and negative.any():
        rank = F.softplus(-(margin[:, positive, None] - margin[:, None, negative])).mean()
    else:
        rank = ce * 0.
    return ce + .25 * rank + float(trust) * kl, dict(ce=ce, rank=rank, kl=kl)
