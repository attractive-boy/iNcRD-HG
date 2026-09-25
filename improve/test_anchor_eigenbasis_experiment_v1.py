"""Verify the reused numerical path and the sole new ablation executor."""
import inspect
import unittest

import numpy as np
import torch

from improve import anchor_preconditioning_experiment_v1 as parent
from improve import anchor_eigenbasis_experiment_v1 as experiment
from improve import eigenbasis_adam_v1 as optimizer
from improve import eigenbasis_graph_v1 as core
from improve import preconditioned_graph_v1 as old_core
from improve.test_nested_anchor_v2 import fixture


def protocol(old=False):
    module = old_core if old else core
    parameters = core.parameters()
    parameters.update(width=8, heads=2)
    return dict(smoke=True, family='eigenbasis_adam',
        implementation_module=module.__name__.split('.')[-1],
        source_sha256={module.__name__.split('.')[-1]+'.py': experiment.common.base.digest(module.__file__)},
        core_parameters=parameters, optimizer_parameters=optimizer.parameters(),
        kinds=list(module.KINDS), epochs=12, check_interval=20, weight_decay=.001, strong_weight=.1)


class EigenbasisExecutorTests(unittest.TestCase):
    def test_training_and_prediction_function_bodies_are_unchanged(self):
        for name in ('fit', 'predict', 'cpu_copy', 'run'):
            self.assertEqual(inspect.getsource(getattr(experiment, name)),
                             inspect.getsource(getattr(parent, name)))

    def test_parent_full_and_adamw_cpu_graph_trajectories_match(self):
        a, _ = fixture()
        a['anchor_oof'] = np.zeros_like(a['y_train'], dtype=np.float64)
        for kind, old_kind in (('full', 'soap'), ('adamw', 'adamw')):
            new, _, trace, detail = experiment.fit(a, protocol(), kind, .001, 17, 'cpu', fixed_epochs=12)
            old, _, old_trace, _ = parent.fit(a, protocol(True), old_kind, .001, 17, 'cpu', fixed_epochs=12)
            for key, value in new.state_dict().items():
                torch.testing.assert_close(value, old.state_dict()[key], atol=0, rtol=0)
            self.assertEqual(trace, old_trace)
            self.assertEqual(detail['gradient_evaluations'], 13)

    def test_separable_training_boundary_budget_and_identity(self):
        a, _ = fixture()
        a['anchor_oof'] = np.zeros_like(a['y_train'], dtype=np.float64)
        p = protocol()
        with self.assertRaisesRegex(ValueError, 'training arrays only'):
            experiment.fit(dict(a, y_val=np.ones(3)), p, 'separable_variance', .001, 17, 'cpu', fixed_epochs=1)
        with self.assertRaisesRegex(ValueError, 'cuda:1'):
            experiment.fit(a, p, 'separable_variance', .001, 17, 'cuda:0', fixed_epochs=1)
        model, _, trace, detail = experiment.fit(a, p, 'separable_variance', .001, 17, 'cpu', fixed_epochs=12)
        self.assertEqual(detail['training_steps'], 12)
        self.assertEqual(detail['gradient_evaluations'], 13)
        self.assertEqual(trace[-1]['basis_updates_min'], 1)
        self.assertTrue(all(g['variance_representation'] == 'separable_row_column'
                            for g in model.optimization_final_state['param_groups']))
        a['pair_offset'] = np.array([0, 14])
        model, _, trace, detail = experiment.fit(a, p, 'separable_variance', .001, 17, 'cpu', fixed_epochs=0)
        self.assertEqual(trace, [])
        self.assertFalse(detail['trained_neural_model'])
        self.assertIsNone(model.optimization_final_state)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
