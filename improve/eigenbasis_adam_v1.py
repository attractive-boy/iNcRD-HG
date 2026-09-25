"""SOAP versus separable second moments in the same learned eigenbasis.

This is a separately declared follow-up to the failed KL-Shampoo study.
SOAP/AdamW trajectories are unchanged and may be explicitly adopted. The
only new optimizer computes the rank-one row/column approximation of SOAP's
second moment for its denominator, corresponding to the paper's Adafactor
comparison. Full second-moment state is retained to match update bookkeeping.
"""
import math

import torch

from improve import kronecker_preconditioning_v1 as parent

KINDS = ('full', 'separable_variance', 'adamw')
parameters = parent.parameters
matrix_parameter_names = parent.matrix_parameter_names


def separable_second_moment(variance):
    if variance.ndim != 2:
        raise ValueError('Expected a matrix second moment')
    total = variance.sum().clamp_min(torch.finfo(variance.dtype).tiny)
    return variance.sum(1, keepdim=True) @ variance.sum(0, keepdim=True) / total


class SeparableBasisAdam(parent.KroneckerOptimizer):
    def __init__(self, model, lr, weight_decay, **recipe):
        super().__init__(model, lr=lr, weight_decay=weight_decay, mode='soap', **recipe)
        for group in self.param_groups:
            group['variance_representation'] = 'separable_row_column'

    def _soap_direction(self, gradient, state, group):
        if group['variance_representation'] != 'separable_row_column':
            raise ValueError('Changed variance representation')
        left, right = state['left_basis'], state['right_basis']
        beta1, beta2 = group['betas']
        projected = left.T @ gradient @ right
        state['exp_avg_sq'].mul_(beta2).addcmul_(projected, projected, value=1-beta2)
        denominator = separable_second_moment(state['exp_avg_sq']).sqrt() + group['eps']
        correction = math.sqrt(1-beta2**state['step']) / (1-beta1**state['step'])
        direction = (left @ ((left.T @ state['exp_avg'] @ right) / denominator) @ right.T) * correction
        state['left_covariance'].lerp_(gradient @ gradient.T, 1-beta2)
        state['right_covariance'].lerp_(gradient.T @ gradient, 1-beta2)
        if state['step'] % group['precondition_frequency'] == 0:
            for axis, side in enumerate(('left', 'right')):
                basis, covariance = state[side+'_basis'], state[side+'_covariance']
                order = torch.argsort(torch.diag(basis.T @ covariance @ basis), descending=True)
                state['exp_avg_sq'] = state['exp_avg_sq'].index_select(axis, order)
                state[side+'_basis'] = torch.linalg.qr(covariance @ basis[:, order]).Q
            state['basis_updates'] += 1
        return direction


def optimizer(model, kind, lr, weight_decay, recipe):
    if kind == 'separable_variance':
        return SeparableBasisAdam(model, lr=lr, weight_decay=weight_decay, **recipe)
    if kind not in ('full', 'adamw'):
        raise ValueError('Undeclared eigenbasis optimizer')
    return parent.optimizer(model, 'soap' if kind == 'full' else 'adamw', lr, weight_decay, recipe)
