"""Attribute-conditioned binary flow completion, inspired by FlowCF (SIGIR 2025).

This independently implemented adapter follows the paper's Bernoulli path and
binary Euler readout, not its RecBole data pipeline. Unlisted pairs are implicit
reconstruction zeros, never asserted to be experimentally verified negatives.
"""
import math

import numpy as np
import torch
from torch import nn


def support_and_mask(pairs, labels, folds, fold, shape, evaluation_pairs):
    """OOF outcomes affect neither the support/prior nor reconstruction targets."""
    support = np.zeros(shape, dtype=np.float32)
    use = (folds != fold) & (labels == 1)
    q = pairs[:, use]
    support[q[0], q[1]] = 1.
    eligible = np.ones(shape, dtype=bool)
    for query in (pairs[:, folds == fold], evaluation_pairs):
        eligible[query[0], query[1]] = False
        if support[query[0], query[1]].any():
            raise ValueError('A query outcome is present in the flow support')
    return support, eligible


def interpolated_state(target, prior_uniform, mask_uniform, t, kind):
    """Use common random numbers for matched discrete/continuous controls."""
    frequency = target.mean(0, keepdim=True)
    if kind == 'homogeneous':
        frequency = frequency.mean().expand_as(frequency)
    source = (prior_uniform < frequency).to(target.dtype)
    if kind == 'continuous':
        return t * target + (1 - t) * source
    if kind not in ('discrete', 'homogeneous'):
        raise ValueError('Unknown interpolation')
    return torch.where(mask_uniform < t, target, source)


def binary_euler(state, prediction, observed, time, delta):
    if not 0 <= time < 1 or not 0 < delta <= 1 - time + 1e-8:
        raise ValueError('Invalid Euler time interval')
    probability = state + delta * (prediction - state) / (1 - time)
    return torch.logical_or(probability >= .5, observed.bool()).to(state.dtype)


class BinaryFlowModel(nn.Module):
    def __init__(self, items, attribute_dim, width=128, time_dim=32):
        super().__init__()
        self.items, self.time_dim = items, time_dim
        self.time_projection = nn.Linear(time_dim, time_dim)
        dimensions = (items + attribute_dim + time_dim, width, width, items)
        layers = []
        for i in range(len(dimensions) - 1):
            layers.extend([nn.Dropout(.1), nn.Linear(dimensions[i], dimensions[i + 1])])
            if i < len(dimensions) - 2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, state, attributes, time):
        half = self.time_dim // 2
        frequency = torch.exp(-math.log(10000) * torch.arange(
            half, device=state.device, dtype=state.dtype) / half) * (2 * math.pi)
        angle = time[:, None] * frequency
        temporal = self.time_projection(torch.cat([angle.cos(), angle.sin()], dim=1))
        return self.network(torch.cat([state, attributes, temporal], dim=1))


@torch.no_grad()
def predict(model, support, attributes, kind, n_steps=10, sampling_steps=2,
            rows=None, batch=128):
    """Each row is independent; the support graph is fixed, not query-built."""
    model.eval()
    if rows is None:
        rows = torch.arange(len(support), device=support.device)
    else:
        rows = torch.as_tensor(np.ascontiguousarray(rows), device=support.device)
    result = []
    for begin in range(0, len(rows), batch):
        indices = rows[begin:begin + batch]
        observed = support[indices]
        state = observed.clone()
        for step in range(n_steps - sampling_steps, n_steps):
            time = step / n_steps
            t = torch.full((len(indices),), time, device=support.device)
            output = model(state, attributes[indices], t)
            if step < n_steps - 1:
                if kind == 'continuous':
                    state = state + (output - state) / (n_steps * (1 - time))
                    state = torch.where(observed.bool(), observed, state)
                else:
                    state = binary_euler(state, output, observed, time, 1 / n_steps)
        result.append(output.cpu().numpy())
    return np.concatenate(result)
