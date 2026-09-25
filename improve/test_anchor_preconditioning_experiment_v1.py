"""Training boundaries and actual optimizer budgets, using a small CPU graph."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from improve import anchor_preconditioning_experiment_v1 as experiment
from improve import preconditioned_graph_v1 as core
from improve import kronecker_preconditioning_v1 as optimizer
from improve.test_nested_anchor_v2 import fixture


def protocol():
    parameters = core.parameters()
    parameters.update(width=8, heads=2, dropout=0.)
    return dict(smoke=True, family='kronecker_preconditioning',
        implementation_module='preconditioned_graph_v1',
        source_sha256={'preconditioned_graph_v1.py': experiment.common.base.digest(core.__file__)},
        core_parameters=parameters, optimizer_parameters=optimizer.parameters(),
        kinds=list(core.KINDS), epochs=12, check_interval=20, weight_decay=.001, strong_weight=.1)


class PreconditioningExecutorTests(unittest.TestCase):
    def test_training_boundaries_device_and_no_all_device_seeding(self):
        a, _ = fixture()
        a['anchor_oof'] = np.zeros_like(a['y_train'], dtype=np.float64)
        contaminated = dict(a, y_val=np.ones(3))
        with self.assertRaisesRegex(ValueError, 'training arrays only'):
            experiment.fit(contaminated, protocol(), 'full', .001, 1, 'cpu', fixed_epochs=1)
        with self.assertRaisesRegex(ValueError, 'training arrays only'):
            experiment.setup(contaminated, protocol(), 'full', 'cpu')
        with self.assertRaisesRegex(ValueError, 'cuda:1'):
            experiment.fit(a, protocol(), 'full', .001, 1, 'cuda:0', fixed_epochs=1)
        with patch.object(torch, 'manual_seed', side_effect=AssertionError('all-device seeding forbidden')):
            _, _, _, detail = experiment.fit(a, protocol(), 'full', .001, 1, 'cpu', fixed_epochs=1)
        self.assertFalse(detail['all_device_seeding_used'])

    def test_matched_update_budget_saved_state_and_graphs(self):
        a, _ = fixture()
        a['anchor_oof'] = np.zeros_like(a['y_train'], dtype=np.float64)
        p = protocol()
        first_loss, graph_reference = None, None
        for kind in core.KINDS:
            model, value, trace, detail = experiment.fit(a, p, kind, .001, 17, 'cpu', fixed_epochs=12)
            self.assertIsNone(value)
            self.assertTrue(detail['trained_neural_model'])
            self.assertEqual(detail['gradient_evaluations'], 13)
            self.assertEqual(detail['training_steps'], 12)
            self.assertEqual([r['epoch'] for r in trace], list(range(13)))
            self.assertTrue(trace[0]['initialization_only'])
            self.assertFalse(any('anchor_composed' in r for r in trace))
            if first_loss is None:
                first_loss = trace[0]['loss']
            self.assertEqual(trace[0]['loss'], first_loss)
            for row in trace:
                self.assertEqual(row['parameter_updates_min'], row['epoch'])
                self.assertEqual(row['omitted_fold'], max(0, row['epoch']-1) % 5)
            for state in model.optimization_final_state['state'].values():
                self.assertEqual(state['step'], 12)
                self.assertEqual(state['gradient_calls'], 13)
            _, _, graphs, _ = experiment.setup(a, p, kind, 'cpu')
            if graph_reference is None:
                graph_reference = graphs
            for graph, reference in zip(graphs, graph_reference):
                for x, y in zip(graph, reference):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_lnc_identity_never_initializes_optimizer_or_constructs_graph(self):
        a, _ = fixture()
        a.update(anchor_oof=np.zeros_like(a['y_train'], dtype=np.float64), pair_offset=np.array([0, 14]))
        for kind in core.KINDS:
            with patch.object(optimizer, 'optimizer', side_effect=AssertionError('Identity has no optimizer')):
                model, _, trace, detail = experiment.fit(a, protocol(), kind, .001, 17, 'cpu', fixed_epochs=0)
            self.assertEqual(model.state_dict(), {})
            self.assertIsNone(model.optimization_final_state)
            self.assertEqual(trace, [])
            self.assertFalse(detail['trained_neural_model'])
            self.assertEqual(detail['gradient_evaluations'], 0)
            _, xs, graphs, _ = experiment.setup(a, protocol(), kind, 'cpu')
            self.assertEqual(graphs, [None]*5)
            scores = experiment.predict(model, xs, graphs, torch.tensor([[0, 1], [14, 15]]), protocol())
            torch.testing.assert_close(scores, torch.zeros(2, dtype=torch.float64), atol=0, rtol=0)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
