"""STORM-inspired, training-only loss rescaling on query-masked graphs.

The task predictor is unchanged ordinary typed attention. The new mechanism is
the two-path meta update of a loss rescaler. No rescaler or labels are used at
prediction time. See STORM_ADAPTATION_V1.md for the explicit local adaptations.
"""
from collections import OrderedDict
from statistics import NormalDist

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import nested_anchor_inputs_v2 as nested
from improve import sharpness_graph_v1 as ordinary

KINDS = ('full', 'no_meta_gradient', 'no_outer_rescaling', 'uniform')
TRAIN_KEYS = nested.TRAIN_KEYS | {'anchor_oof'}
identity_task = ordinary.identity_task


def parameters():
    return dict(**ordinary.parameters(), dropout_passes=3, feature_count=19,
        rescaler_learning_rate=.01, rescaler_weight_decay=0.,
        weight_multiplier=2., memory_batches_per_class=1,
        feature_epsilon=1e-8, adam_betas=[.9, .999], adam_epsilon=1e-8,
        gradient_clip=5., episode='disjoint random halves of one omitted fold',
        rescaler_batchnorm='each class network sees the whole batch before label selection',
        memory_scope='separate omitted-fold and inner/meta role; training samples only')


def episode_indices(ids, rng):
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) < 4 or len(np.unique(ids)) != len(ids):
        raise ValueError('An episode needs at least four distinct training queries')
    shuffled = rng.permutation(ids)
    middle = len(shuffled) // 2
    return shuffled[:middle], shuffled[middle:]


class ClassRescaler(nn.Module):
    """Released-code BN population and output component, uniform initialization.

    Non-affine BN removes a common learned logit shift. It is not a proof that
    label-selected outputs cannot become small. We record their actual ranges.
    Running statistics are unnecessary because this module is training-only.
    """
    def __init__(self, features=19):
        super().__init__()
        self.networks = nn.ModuleList([nn.Sequential(
            nn.Linear(features, features),
            nn.BatchNorm1d(features, affine=False, track_running_stats=False),
            nn.ReLU(), nn.Linear(features, 2),
            nn.BatchNorm1d(2, affine=False, track_running_stats=False),
            nn.Softmax(dim=1)) for _ in range(2)])
        # Equal, nonzero output rows give exactly 1/2 while retaining a usable
        # initial derivative through the final batch normalization.
        with torch.no_grad():
            for network in self.networks:
                network[3].weight[1].copy_(network[3].weight[0])
                network[3].bias[1].copy_(network[3].bias[0])

    def forward(self, features, labels):
        if len(features) < 2 or features.shape != (len(labels), 19):
            raise ValueError('Expected at least two rows of 19 detached features')
        if features.requires_grad:
            raise ValueError('STORM features must be detached from the task predictor')
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError('Expected binary training labels')
        outputs = torch.stack([net(features)[:, 1] for net in self.networks], dim=1)
        return outputs.gather(1, labels.long()[:, None]).squeeze(1)


class MetaLossGraph(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.rescaler = ClassRescaler()
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()

    def forward(self, xs, graph, pairs):
        return self.backbone(xs, graph, pairs)


def setup(a, p, kind, device):
    if set(a) - TRAIN_KEYS:
        raise ValueError('Meta-loss setup accepts training arrays only')
    if kind not in KINDS:
        raise ValueError('Unregistered meta-loss arm')
    backbone, xs, graphs, folds = ordinary.setup(a, p, 'erm', device)
    if identity_task(a, p):
        return backbone, xs, graphs, folds
    return MetaLossGraph(backbone).to(device), xs, graphs, folds


def _sample_std(values, axis=None, eps=1e-8):
    count = values.size if axis is None else values.shape[axis]
    if count <= 1:
        return eps
    # Center around an actual observation first. Directly averaging three
    # identical FP64 losses can round by one ULP and fabricate a tiny variance;
    # its inverse in KL then overflows the FP32 rescaler. This translation keeps
    # the mathematical sample variance and makes a constant sample exactly zero.
    origin = values.flat[0] if axis is None else np.take(values, [0], axis=axis)
    return np.std(values - origin, axis=axis, ddof=1)


def gaussian_kl(mu, sigma, group_mu, group_sigma):
    # This zero-variance convention is present in the released author utility.
    if sigma == 0. or group_sigma == 0.:
        return 0.
    return float(np.log(group_sigma / sigma)
        + (sigma * sigma + (mu - group_mu)**2) / (2 * group_sigma**2) - .5)


def gaussian_overlap(mu, sigma, group_mu, group_sigma, eps=1e-8):
    return NormalDist(float(mu), max(float(sigma), eps)).overlap(
        NormalDist(float(group_mu), max(float(group_sigma), eps)))


class FeatureMemory:
    """Detached rolling, class-and-agreement statistics from one support context.

    Capacity is one episode batch per target class, as in the released Filter.
    Inner and meta roles have different instances. IDs are kept for boundary
    audits only; they are not rescaler inputs.
    """
    def __init__(self, capacity, eps=1e-8):
        if capacity < 2:
            raise ValueError('Feature memory capacity must be at least two')
        self.capacity, self.eps = int(capacity), float(eps)
        self.rows = {c: np.empty((0, 6), dtype=np.float64) for c in (0, 1)}

    def state_dict(self):
        return dict(capacity=self.capacity, eps=self.eps,
                    rows={str(k): v.copy() for k, v in self.rows.items()})

    def load_state_dict(self, state):
        if state['capacity'] != self.capacity or state['eps'] != self.eps:
            raise ValueError('Feature memory configuration changed')
        self.rows = {int(k): np.asarray(v, dtype=np.float64).copy()
                     for k, v in state['rows'].items()}

    def features(self, logits, losses, labels, ids):
        logits = np.asarray(logits, dtype=np.float64)
        losses = np.asarray(losses, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        ids = np.asarray(ids, dtype=np.int64)
        if logits.shape != losses.shape or logits.ndim != 2 or logits.shape[0] < 2:
            raise ValueError('Expected multiple detached dropout observations')
        if logits.shape[1] != len(labels) or labels.shape != ids.shape or len(labels) > self.capacity:
            raise ValueError('Feature batch exceeds its declared memory capacity')
        if not (np.isfinite(logits).all() and np.isfinite(losses).all()):
            raise FloatingPointError('Nonfinite feature observations')
        if not np.isin(labels, [0, 1]).all():
            raise ValueError('Expected binary training labels')
        # max(sigmoid(z), 1-sigmoid(z)) = sigmoid(abs(z)), stable for all z.
        confidence = 1. / (1. + np.exp(-np.abs(logits)))
        agreements = (logits > 0) == labels[None]
        count = agreements.sum(0)
        agreement = count > logits.shape[0] / 2
        ties = count == logits.shape[0] / 2
        agreement[ties] = agreements[0, ties]
        current = np.column_stack((losses.mean(0), _sample_std(losses, axis=0),
            confidence.mean(0), _sample_std(confidence, axis=0), agreement, ids))
        for c in (0, 1):
            self.rows[c] = np.concatenate((self.rows[c], current[labels == c]))[-self.capacity:]
        result = np.empty((len(labels), 19), dtype=np.float64)
        for c in (0, 1):
            for agree in (False, True):
                selected = (labels == c) & (agreement == agree)
                if not selected.any():
                    continue
                history = self.rows[c][self.rows[c][:, 4] == agree]
                if not len(history):
                    raise AssertionError('Current samples disappeared from their feature memory')
                lm, ls, pm, ps = (history[:, i] for i in range(4))
                lm_mean, ls_mean = float(lm.mean()), float(ls.mean())
                kls = np.array([gaussian_kl(m, s, lm_mean, ls_mean) for m, s in zip(lm, ls)])
                ovls = np.array([gaussian_overlap(m, s, lm_mean, ls_mean, self.eps) for m, s in zip(lm, ls)])
                for i in np.flatnonzero(selected):
                    m, s, prob, prob_sd = current[i, :4]
                    result[i] = [m, lm_mean, _sample_std(lm, eps=self.eps),
                        s, ls_mean, _sample_std(ls, eps=self.eps),
                        prob, float(pm.mean()), _sample_std(pm, eps=self.eps),
                        prob_sd, float(ps.mean()), _sample_std(ps, eps=self.eps),
                        gaussian_kl(m, s, lm_mean, ls_mean), float(kls.mean()), _sample_std(kls, eps=self.eps),
                        gaussian_overlap(m, s, lm_mean, ls_mean, self.eps), float(ovls.mean()),
                        _sample_std(ovls, eps=self.eps), float(agree)]
        result = result.astype(np.float32)
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite STORM input features')
        return result


def differentiable_adamw(params, gradients, state, *, lr, weight_decay,
                         betas=(.9, .999), eps=1e-8, clip=5.):
    """One actual AdamW update, differentiable through the current gradients.

    Moments from prior episodes are detached. A zero second moment gets the
    finite zero subgradient used for inactive coordinates. At nonzero moments
    this is the ordinary AdamW map, including bias correction and clipping.
    """
    if set(params) != set(gradients) or (state and set(state['first']) != set(params)):
        raise ValueError('AdamW parameter names do not match')
    if any(g is None for g in gradients.values()):
        raise ValueError('Task parameter gradient is missing')
    step = int(state.get('step', 0)) + 1
    beta1, beta2 = betas
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gradients.values()]))
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite task gradient before the differentiable AdamW update')
    coefficient = (clip / (norm + 1e-6)).clamp(max=1.)
    updated, first, second = OrderedDict(), {}, {}
    for name, parameter in params.items():
        g = gradients[name] * coefficient
        old_m = state['first'][name].detach() if state else torch.zeros_like(parameter)
        old_v = state['second'][name].detach() if state else torch.zeros_like(parameter)
        m = beta1 * old_m + (1. - beta1) * g
        v = beta2 * old_v + (1. - beta2) * g.square()
        variance = v / (1. - beta2**step)
        positive = variance > 0
        root = torch.where(positive, variance, torch.ones_like(variance)).sqrt()
        root = torch.where(positive, root, torch.zeros_like(root))
        updated[name] = parameter * (1. - lr * weight_decay) - (lr / (1. - beta1**step)) * m / (root + eps)
        first[name], second[name] = m, v
    return updated, dict(step=step, first=first, second=second), norm


def detached_optimizer_state(state):
    return dict(step=state['step'], first={k: v.detach().clone() for k, v in state['first'].items()},
                second={k: v.detach().clone() for k, v in state['second'].items()})


def observed_batch(backbone, params, xs, graph, pairs, anchor, labels, passes):
    """Only the first of three dropout forwards keeps a task gradient graph."""
    all_logits, all_losses = [], []
    for i in range(passes):
        with torch.set_grad_enabled(i == 0):
            residual = torch.func.functional_call(backbone, params, (xs, graph, pairs), strict=True)
            logits = anchor + residual.double()
            losses = F.binary_cross_entropy_with_logits(logits, labels.double(), reduction='none')
        if i == 0:
            differentiable_loss = losses
        all_logits.append(logits.detach().cpu().numpy())
        all_losses.append(losses.detach().cpu().numpy())
    return differentiable_loss, np.stack(all_logits), np.stack(all_losses)


def scaled_weights(rescaler, features, labels, kind, role, multiplier=2.):
    learned = multiplier * rescaler(features, labels)
    if kind == 'uniform' or (kind == 'no_outer_rescaling' and role == 'meta'):
        return torch.ones_like(learned) + learned * 0.
    return learned


def gradient_norm(gradients):
    return float(torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(g.detach().double()) for g in gradients])))
