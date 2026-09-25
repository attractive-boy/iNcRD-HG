"""Common strong-reference executor with enforced, recorded equilibrium checks."""
import argparse

import torch

from improve import anchor_priority_graph_v1 as shared
from improve import equilibrium_graph_v1 as core

common, ROOT, nested = shared.common, shared.ROOT, shared.nested
TRAIN_KEYS = shared.TRAIN_KEYS
prepare, setup, predict, select = shared.prepare, shared.setup, shared.predict, shared.select
BASE_FIT = shared.fit


def fit(*args, **kwargs):
    model, value, trace, detail = BASE_FIT(*args, **kwargs)
    diagnostics = core.solver_summary(model)
    if detail['trained_neural_model']:
        expected = detail['training_steps'] * len(model.factors)
        if diagnostics['training_calls'] != expected:
            raise ValueError('Missing registered propagation calls')
        if model.kind == 'full':
            if diagnostics['backward']['calls'] != expected:
                raise ValueError('Missing implicit adjoint solves')
            for direction in ('forward', 'backward'):
                if diagnostics[direction]['max_relative_residual'] > model.p['solver_tolerance']:
                    raise ValueError('Nonconverged propagation was used')
        with torch.no_grad():
            detail['selected_coupling_max_eigenvalue'] = max(float(torch.linalg.eigvalsh(
                core.normalized_gram(factor.double())).max()) for factor in model.factors)
        if detail['selected_coupling_max_eigenvalue'] > 1. + 1e-12:
            raise ValueError('Coupling bound failed')
    detail['solver_diagnostics'] = diagnostics
    return model, value, trace, detail


def run(out, task, device, stage):
    # The shared writer is unchanged; the extra training diagnostics are part of
    # the new protocol and are saved in its ordinary result.json detail fields.
    previous = shared.fit
    shared.fit = fit
    try:
        return shared.run(out, task, device, stage)
    finally:
        shared.fit = previous


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('screen', 'confirm'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--task', required=True, choices=('MiDrug', 'LncDrug'))
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    run(args.out, args.task, args.device, args.stage)
