"""Two-sided KL-Shampoo, its scalar-coupling ablation, and optimizer controls.

Independent matrix-only adaptation of arXiv:2509.03378v10, equations (5)--(7),
Figure 12, and the pinned author's implementation. SOAP follows Algorithm 3
of arXiv:2409.11321v2, including the author's variance permutation at QR steps.
Only nondegenerate Linear weight matrices receive matrix preconditioning.
Every arm consumes one initialization gradient without changing parameters,
then exactly the registered number of updates. Other parameters use AdamW.
"""
import math

import torch
from torch import nn

KINDS = ('full', 'scalar_coupling', 'soap', 'adamw')


def parameters():
    return dict(betas=[.9, .98], eps=1e-8, precondition_frequency=10,
                init_factor=.1, max_inverse_sqrt=4000,
                curvature_damping=0., normalize_updates=False,
                initialization_gradients=1, weight_decay_before_update=True)


def matrix_parameter_names(model):
    return tuple((name + '.' if name else '') + 'weight' for name, module in model.named_modules()
                 if isinstance(module, nn.Linear) and module.weight.requires_grad
                 and min(module.weight.shape) > 1)


def initial_basis(matrix):
    """Descending eigenvectors; fp64 is retained for mathematical tests."""
    jitter = torch.eye(len(matrix), device=matrix.device, dtype=matrix.dtype) * 1e-30
    try:
        _, basis = torch.linalg.eigh(matrix + jitter)
    except torch.linalg.LinAlgError:
        _, basis = torch.linalg.eigh((matrix + jitter).double())
        basis = basis.to(matrix.dtype)
    return basis.flip(1)


def covariance_targets(gradient, left_basis, right_basis, left_inverse,
                       right_inverse, mode):
    """Simultaneous covariance/eigenvalue targets, using only the OLD factors.

    The scalar ablation replaces each opposite inverse covariance by its
    mean eigenvalue times identity, preserving its average inverse scale.
    """
    if mode not in ('full', 'scalar_coupling'):
        raise ValueError('Unknown covariance-estimation mechanism')
    rows, columns = gradient.shape
    projected = left_basis.T @ gradient @ right_basis
    if mode == 'full':
        right_whitened = (gradient @ right_basis) * right_inverse
        left_whitened = (gradient.T @ left_basis) * left_inverse
        left = right_whitened @ right_whitened.T / columns
        right = left_whitened @ left_whitened.T / rows
        left_diagonal = (projected * right_inverse).square().mean(1)
        right_diagonal = (projected * left_inverse[:, None]).square().mean(0)
    else:
        right_scale = right_inverse.square().mean()
        left_scale = left_inverse.square().mean()
        left = (gradient @ gradient.T) * (right_scale / columns)
        right = (gradient.T @ gradient) * (left_scale / rows)
        left_diagonal = projected.square().mean(1) * right_scale
        right_diagonal = projected.square().mean(0) * left_scale
    return left, right, left_diagonal, right_diagonal


def inverse_eigenvalue_update(inverse, target, beta, maximum):
    eigenvalue = inverse.square().reciprocal()
    eigenvalue.lerp_(target, 1 - beta)
    # Equivalent to clamping inverse square roots, without hiding NaN/Inf.
    cap = max(10, min(inverse.numel(), maximum))
    return eigenvalue.clamp_min(1. / cap ** 2).rsqrt()


class KroneckerOptimizer(torch.optim.Optimizer):
    """Single-state-dict optimizer with explicitly matched warm starts."""
    def __init__(self, model, lr, weight_decay=.001, mode='full', betas=(.9, .98),
                 eps=1e-8, precondition_frequency=10, init_factor=.1,
                 max_inverse_sqrt=4000, curvature_damping=0.,
                 normalize_updates=False, initialization_gradients=1,
                 weight_decay_before_update=True):
        if (mode not in KINDS or lr <= 0 or weight_decay < 0 or eps <= 0
                or len(betas) != 2 or any(not 0 <= b < 1 for b in betas)
                or precondition_frequency < 1 or init_factor <= 0
                or max_inverse_sqrt < 1 or curvature_damping != 0.
                or normalize_updates or initialization_gradients != 1
                or not weight_decay_before_update):
            raise ValueError('Invalid or unsupported frozen optimizer recipe')
        names = set(matrix_parameter_names(model))
        if not names:
            raise ValueError('No nondegenerate linear matrices to precondition')
        self.matrix_names = tuple(sorted(names))
        matrices, others = [], []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                (matrices if name in names else others).append(parameter)
        groups = [dict(params=matrices, matrix_parameter=True),
                  dict(params=others, matrix_parameter=False)]
        defaults = dict(lr=lr, weight_decay=weight_decay, mode=mode, betas=tuple(betas),
                        eps=eps, precondition_frequency=precondition_frequency,
                        init_factor=init_factor, max_inverse_sqrt=max_inverse_sqrt)
        super().__init__(groups, defaults)

    def _initialize(self, gradient, state, group):
        state.update(step=0, gradient_calls=1, exp_avg=torch.zeros_like(gradient))
        mode = group['mode']
        if not group['matrix_parameter'] or mode == 'adamw':
            state['exp_avg_sq'] = torch.zeros_like(gradient)
            return
        rows, columns = gradient.shape
        beta = group['betas'][1]
        divisor_left, divisor_right = ((1, 1) if mode == 'soap' else (columns, rows))
        state['left_covariance'] = (1 - beta) * (gradient @ gradient.T) / divisor_left
        state['right_covariance'] = (1 - beta) * (gradient.T @ gradient) / divisor_right
        state['left_basis'] = initial_basis(state['left_covariance'])
        state['right_basis'] = initial_basis(state['right_covariance'])
        state['basis_updates'] = 0
        if mode == 'soap':
            state['exp_avg_sq'] = torch.zeros_like(gradient)
        else:
            state['left_inverse'] = gradient.new_full((rows,), group['init_factor'] ** -.5)
            state['right_inverse'] = gradient.new_full((columns,), group['init_factor'] ** -.5)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Validate all gradients before mutating any parameter/state.
        for group in self.param_groups:
            for parameter in group['params']:
                g = parameter.grad
                if g is not None and (g.is_sparse or g.dtype not in (torch.float32, torch.float64)
                                      or not torch.isfinite(g).all()):
                    raise FloatingPointError('Expected finite dense fp32/fp64 gradients')
        for group in self.param_groups:
            for parameter in group['params']:
                gradient = parameter.grad
                if gradient is None:
                    continue
                state = self.state[parameter]
                if not state:
                    self._initialize(gradient, state, group)
                    continue
                state['step'] += 1
                state['gradient_calls'] += 1
                beta1, beta2 = group['betas']
                state['exp_avg'].lerp_(gradient, 1 - beta1)
                matrix = group['matrix_parameter'] and group['mode'] != 'adamw'
                if not matrix:
                    state['exp_avg_sq'].mul_(beta2).addcmul_(gradient, gradient, value=1-beta2)
                    denominator = state['exp_avg_sq'].sqrt() / math.sqrt(1-beta2**state['step'])
                    denominator.add_(group['eps'])
                    direction = (state['exp_avg'] / (1-beta1**state['step'])) / denominator
                elif group['mode'] == 'soap':
                    direction = self._soap_direction(gradient, state, group)
                else:
                    left, right = state['left_basis'], state['right_basis']
                    scale = state['left_inverse'][:, None] * state['right_inverse'][None, :]
                    scale = scale / (1 + group['eps'] * scale)
                    direction = left @ ((left.T @ state['exp_avg'] @ right) * scale) @ right.T
                    self._update_kl(gradient, state, group)
                parameter.mul_(1-group['lr'] * group['weight_decay'])
                parameter.add_(direction, alpha=-group['lr'])
        return loss

    def _update_kl(self, gradient, state, group):
        beta = group['betas'][1]
        left, right, diagonal_left, diagonal_right = covariance_targets(
            gradient, state['left_basis'], state['right_basis'],
            state['left_inverse'], state['right_inverse'], group['mode'])
        state['left_covariance'].lerp_(left, 1-beta)
        state['right_covariance'].lerp_(right, 1-beta)
        state['left_inverse'] = inverse_eigenvalue_update(
            state['left_inverse'], diagonal_left, beta, group['max_inverse_sqrt'])
        state['right_inverse'] = inverse_eigenvalue_update(
            state['right_inverse'], diagonal_right, beta, group['max_inverse_sqrt'])
        if state['step'] % group['precondition_frequency'] == 0:
            for side in ('left', 'right'):
                state[side + '_basis'] = torch.linalg.qr(
                    state[side + '_covariance'] @ state[side + '_basis']).Q
            state['basis_updates'] += 1

    def _soap_direction(self, gradient, state, group):
        left, right = state['left_basis'], state['right_basis']
        beta1, beta2 = group['betas']
        projected = left.T @ gradient @ right
        state['exp_avg_sq'].mul_(beta2).addcmul_(projected, projected, value=1-beta2)
        denominator = state['exp_avg_sq'].sqrt() + group['eps']
        correction = math.sqrt(1-beta2**state['step']) / (1-beta1**state['step'])
        direction = (left @ ((left.T @ state['exp_avg'] @ right) / denominator) @ right.T) * correction
        state['left_covariance'].lerp_(gradient @ gradient.T, 1-beta2)
        state['right_covariance'].lerp_(gradient.T @ gradient, 1-beta2)
        if state['step'] % group['precondition_frequency'] == 0:
            # The author's approximation permutes V; it does not rotate a full
            # covariance tensor when the basis changes. Momentum stays original.
            for axis, side in enumerate(('left', 'right')):
                basis, covariance = state[side + '_basis'], state[side + '_covariance']
                order = torch.argsort(torch.diag(basis.T @ covariance @ basis), descending=True)
                state['exp_avg_sq'] = state['exp_avg_sq'].index_select(axis, order)
                state[side + '_basis'] = torch.linalg.qr(covariance @ basis[:, order]).Q
            state['basis_updates'] += 1
        return direction

    def diagnostics(self):
        calls, steps, updates, inverse_max, finite = [], [], [], [], True
        bytes_used, matrix_states = 0, 0
        for group in self.param_groups:
            for parameter in group['params']:
                state = self.state[parameter]
                if not state:
                    continue
                calls.append(state['gradient_calls']); steps.append(state['step'])
                if 'left_basis' in state:
                    matrix_states += 1
                    updates.append(state['basis_updates'])
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        finite = finite and bool(torch.isfinite(value).all())
                        bytes_used += value.numel() * value.element_size()
                        if key.endswith('_inverse'):
                            inverse_max.append(float(value.max()))
        return dict(parameter_gradient_calls_min=min(calls), parameter_gradient_calls_max=max(calls),
                    parameter_updates_min=min(steps), parameter_updates_max=max(steps),
                    matrix_states=matrix_states, basis_updates_min=min(updates, default=0),
                    basis_updates_max=max(updates, default=0), state_bytes=bytes_used,
                    inverse_sqrt_max=max(inverse_max, default=0.), state_finite=finite)


def optimizer(model, kind, lr, weight_decay, recipe):
    return KroneckerOptimizer(model, lr=lr, weight_decay=weight_decay, mode=kind, **recipe)
