"""Muon matrix updates and a norm-matched momentum-direction control.

Independent implementation of the five-step Moonlight recipe. The finite
quintic iteration approximates a polar factor; it is not an exact SVD or
orthogonality constraint on the model weights.
"""
import math

import torch
from torch import nn


def newton_schulz(matrix, steps=5):
    if matrix.ndim != 2 or steps < 1:
        raise ValueError('Muon requires a matrix and a positive iteration count')
    # Small graph matrices use fp32 (fp64 in numerical checks), without compile.
    x = matrix if matrix.dtype in (torch.float32, torch.float64) else matrix.float()
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.T
        x = 3.4445 * x + (-4.7750 * gram + 2.0315 * (gram @ gram)) @ x
    return x.T if transpose else x


def matrix_direction(momentum_direction, mode, steps=5):
    polar = newton_schulz(momentum_direction, steps)
    if mode == 'full':
        return polar
    if mode == 'frobenius':
        # Keep the exact finite-NS Frobenius norm and compute budget; replace
        # only its singular-direction reweighting by the original direction.
        norm = momentum_direction.norm().clamp_min(torch.finfo(momentum_direction.dtype).tiny)
        return momentum_direction * (polar.norm() / norm)
    raise ValueError('Unregistered matrix-direction mode')


def matrix_parameter_names(model):
    return tuple(name + '.weight' for name, module in model.named_modules()
                 if isinstance(module, nn.Linear) and module.weight.requires_grad
                 and min(module.weight.shape) > 1)


class SpectralMuon(torch.optim.Optimizer):
    def __init__(self, model, lr, weight_decay=.001, mode='full', momentum=.95,
                 ns_steps=5, update_scale=.2, betas=(.9, .999), eps=1e-8):
        if mode not in ('full', 'frobenius'):
            raise ValueError('Unregistered Muon control')
        if not 0 <= momentum < 1 or not 0 <= weight_decay or lr <= 0 or update_scale <= 0:
            raise ValueError('Invalid optimizer configuration')
        names = set(matrix_parameter_names(model))
        if not names:
            raise ValueError('No hidden linear matrices were selected')
        matrix, other = [], []
        for name, value in model.named_parameters():
            if value.requires_grad:
                (matrix if name in names else other).append(value)
        self.matrix_names = tuple(sorted(names))
        groups = [dict(params=matrix, matrix_update=True), dict(params=other, matrix_update=False)]
        defaults = dict(lr=lr, weight_decay=weight_decay, mode=mode, momentum=momentum,
                        ns_steps=ns_steps, update_scale=update_scale, betas=betas, eps=eps)
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Detect invalid gradients before changing any parameter or optimizer state.
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None and (parameter.grad.is_sparse or not torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError('Muon requires finite dense gradients')
        for group in self.param_groups:
            lr, decay = group['lr'], group['weight_decay']
            for parameter in group['params']:
                grad = parameter.grad
                if grad is None:
                    continue
                state = self.state[parameter]
                if group['matrix_update']:
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(parameter)
                    buffer = state['momentum_buffer']
                    buffer.mul_(group['momentum']).add_(grad)
                    direction = grad.add(buffer, alpha=group['momentum'])
                    update = matrix_direction(direction, group['mode'], group['ns_steps'])
                    scale = group['update_scale'] * math.sqrt(max(parameter.shape))
                    parameter.mul_(1 - lr * decay)
                    parameter.add_(update, alpha=-lr * scale)
                else:
                    if not state:
                        state.update(step=0, exp_avg=torch.zeros_like(parameter),
                                     exp_avg_sq=torch.zeros_like(parameter))
                    state['step'] += 1
                    b1, b2 = group['betas']
                    state['exp_avg'].lerp_(grad, 1 - b1)
                    state['exp_avg_sq'].mul_(b2).addcmul_(grad, grad, value=1 - b2)
                    correction1 = 1 - b1 ** state['step']
                    correction2 = 1 - b2 ** state['step']
                    denominator = state['exp_avg_sq'].sqrt().div_(math.sqrt(correction2)).add_(group['eps'])
                    parameter.mul_(1 - lr * decay)
                    parameter.addcdiv_(state['exp_avg'], denominator, value=-lr / correction1)
        return loss


def optimizer(model, kind, lr, weight_decay, parameters):
    if kind == 'adamw':
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=lr, weight_decay=weight_decay,
                                betas=tuple(parameters['betas']), eps=parameters['eps'])
    return SpectralMuon(model, lr, weight_decay=weight_decay, mode=kind, **parameters)


def parameters():
    return dict(momentum=.95, ns_steps=5, update_scale=.2, betas=[.9, .999], eps=1e-8)
