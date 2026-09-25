"""Two-way tail-risk blocks and proximal stages for masked graph residuals.

Independent adaptation of STACO2 (TMLR 09/2025), equations (9)-(12) and
Algorithm 2. The executor adds BCE and uses AdamW, so the original solver's
convergence theorem does not apply. No downloaded author code is executed.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch.nn import functional as F

from improve import sharpness_graph_v1 as backbone

KINDS = ('full', 'no_proximal', 'bce')
identity_task = backbone.identity_task


def parameters():
    return dict(backbone.parameters(), theta_positive=.5, theta_negative=.5,
        margin=.5, dual_step=.5, threshold_step=.1, outer_threshold_step=.01,
        proximal_gamma=100., inner_steps=50, positive_batch=128, negative_batch=256,
        risk_weight=.1, threshold_initialization='training_oof_fractional_tail_quantiles')


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Unregistered tail-risk arm')
    return backbone.setup(a, p, 'erm', device)


def pair_loss(positive, negative, margin):
    return F.relu(margin + negative[None, :] - positive[:, None]).square()


def tail_threshold(values, fraction):
    """Minimizer of empirical fractional CVaR, including noninteger tail size."""
    if not 0 < fraction <= 1 or values.shape[-1] < 1:
        raise ValueError('Tail fraction and nonempty sample are required')
    count = values.shape[-1]
    k = int(math.ceil(count * fraction))
    return values.kthvalue(count-k+1, dim=-1).values


def cvar(values, threshold, fraction):
    return threshold + F.relu(values-threshold[..., None]).mean(-1) / fraction


@dataclass
class RiskBatch:
    positive_global: torch.Tensor
    positive_local: torch.Tensor
    negative_dual: torch.Tensor
    negative_primal: torch.Tensor


class TailState:
    """Training-row state; validation arrays are not accepted or referenced."""
    def __init__(self, labels, folds, anchor_oof, p, seed, device):
        labels = np.asarray(labels)
        folds = np.asarray(folds)
        if (labels.ndim != 1 or labels.shape != folds.shape or labels.shape != np.shape(anchor_oof)
                or not np.isin(labels, [0, 1]).all() or not np.isfinite(anchor_oof).all()):
            raise ValueError('Invalid training ranking inputs')
        for key in ('theta_positive', 'theta_negative'):
            if not 0 < p[key] <= 1:
                raise ValueError('Invalid tail fraction')
        if min(p[k] for k in ('dual_step', 'threshold_step', 'outer_threshold_step',
                'proximal_gamma', 'inner_steps', 'positive_batch', 'negative_batch')) <= 0:
            raise ValueError('Positive update parameters are required')
        self.p, self.device = p, device
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.s = torch.ones(len(labels), device=device, dtype=torch.float64)
        self.y = torch.ones_like(self.s)
        self.positive = torch.as_tensor(np.flatnonzero(labels == 1), device=device)
        self.fold_rows, self.positive_local, self.negative_local = [], [], []
        probability = torch.as_tensor(anchor_oof, device=device, dtype=torch.float64).sigmoid()
        risk = []
        for fold in range(p['folds']):
            rows = np.flatnonzero(folds == fold)
            positive = np.flatnonzero(labels[rows] == 1)
            negative = np.flatnonzero(labels[rows] == 0)
            if not len(positive) or not len(negative):
                raise ValueError('Every query fold must contain both classes')
            self.fold_rows.append(torch.as_tensor(rows, device=device))
            self.positive_local.append(torch.as_tensor(positive, device='cpu'))
            self.negative_local.append(torch.as_tensor(negative, device='cpu'))
            ip = torch.as_tensor(rows[positive], device=device)
            inn = torch.as_tensor(rows[negative], device=device)
            losses = pair_loss(probability[ip], probability[inn], p['margin'])
            threshold = tail_threshold(losses, p['theta_negative'])
            self.s[ip] = threshold
            risk.append(cvar(losses, threshold, p['theta_negative']))
        self.prime = tail_threshold(torch.cat(risk), p['theta_positive'])
        self.completed_stages = 0
        self.updates = 0
        self.center_parameters = None

    def sample(self, fold):
        def draw(pool, count):
            index = torch.randperm(len(pool), generator=self.generator)[:min(len(pool), count)]
            return pool[index].to(self.device)
        pos = draw(self.positive_local[fold], self.p['positive_batch'])
        # Independent draws may overlap; enforcing disjointness would couple them.
        negative_dual = draw(self.negative_local[fold], self.p['negative_batch'])
        negative_primal = draw(self.negative_local[fold], self.p['negative_batch'])
        return RiskBatch(self.fold_rows[fold][pos], pos, negative_dual, negative_primal)

    @torch.no_grad()
    def begin_stage(self, model):
        self.y[self.positive] = 1.
        self.center_s = self.s.clone()
        self.center_parameters = [v.detach().clone() for v in model.parameters() if v.requires_grad]
        self.sum_parameters = [torch.zeros_like(v) for v in self.center_parameters]
        self.sum_s = torch.zeros_like(self.s)
        self.sum_prime = torch.zeros_like(self.prime)
        self.inner_count = 0

    def block_objective(self, probability, batch, kind):
        if kind not in KINDS or self.center_parameters is None:
            raise ValueError('A declared arm and active proximal stage are required')
        p = self.p
        rows = batch.positive_global
        threshold = self.s[rows].detach().clone()
        prime = self.prime.detach().clone()
        positive = probability[batch.positive_local]
        with torch.no_grad():
            loss_dual = pair_loss(positive.detach(), probability[batch.negative_dual].detach(), p['margin'])
            risk_dual = cvar(loss_dual, threshold, p['theta_negative'])
            new_y = (self.y[rows] + p['dual_step'] * (risk_dual-prime) / p['theta_positive']).clamp(0, 1)
            self.y[rows] = new_y
        loss_primal = pair_loss(positive, probability[batch.negative_primal], p['margin'])
        risk_primal = cvar(loss_primal, threshold, p['theta_negative'])
        objective = prime + (new_y * (risk_primal-prime)).mean() / p['theta_positive']
        with torch.no_grad():
            active = loss_primal.detach() > threshold[:, None]
            derivative_s = 1 - active.double().mean(-1) / p['theta_negative']
            proximal_s = ((threshold-self.center_s[rows]) / p['proximal_gamma']
                          if kind != 'no_proximal' else torch.zeros_like(threshold))
            # Equation (10), including its printed 1 / theta_positive scaling.
            self.s[rows] = threshold - p['threshold_step'] / p['theta_positive'] * (
                new_y * derivative_s + proximal_s)
            self.prime = prime - p['outer_threshold_step'] * (1-new_y.mean()/p['theta_positive'])
            self.updates += 1
        detail = dict(sampled_positives=len(rows), sampled_negatives=len(batch.negative_primal),
            dual_batch_equals_primal_batch=bool(torch.equal(batch.negative_dual, batch.negative_primal)),
            mean_selected_dual=float(new_y.mean()), min_selected_dual=float(new_y.min()),
            max_selected_dual=float(new_y.max()), active_negative_fraction=float(active.double().mean()),
            outer_threshold=float(self.prime), mean_positive_threshold=float(self.s[self.positive].mean()))
        return objective, detail

    def parameter_penalty(self, model, kind):
        values = [v for v in model.parameters() if v.requires_grad]
        if kind == 'no_proximal':
            return values[0].sum() * 0.
        squared = sum((v-center).square().sum() for v, center in zip(values, self.center_parameters))
        return squared / (2*self.p['theta_positive']*self.p['proximal_gamma'])

    @torch.no_grad()
    def accumulate(self, model, finish=False):
        values = [v for v in model.parameters() if v.requires_grad]
        self.inner_count += 1
        for total, value in zip(self.sum_parameters, values):
            total.add_(value)
        self.sum_s.add_(self.s)
        self.sum_prime.add_(self.prime)
        completed = self.inner_count == self.p['inner_steps'] or finish
        if completed:
            for value, total in zip(values, self.sum_parameters):
                value.copy_(total/self.inner_count)
            self.s.copy_(self.sum_s/self.inner_count)
            self.prime.copy_(self.sum_prime/self.inner_count)
            self.completed_stages += 1
        return completed

    def snapshot(self):
        return dict(s=self.s.detach().cpu().clone(), y=self.y.detach().cpu().clone(),
            s_prime=self.prime.detach().cpu().clone(), positive_rows=self.positive.cpu().clone(),
            rng_state=self.generator.get_state(), updates=self.updates,
            completed_stages=self.completed_stages, inner_count=self.inner_count,
            inference_uses_training_risk_state=False)
