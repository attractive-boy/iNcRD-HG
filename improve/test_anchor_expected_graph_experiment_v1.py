"""Fixed-training isolation and stage refreshes for auxiliary PU self-training."""
import unittest

import numpy as np
import torch

from improve import anchor_expected_graph_experiment_v1 as experiment
from improve import expected_graph_selftraining_v1 as core
from improve.test_nested_anchor_v2 import fixture


def protocol():
    parameters = core.parameters()
    parameters.update(width=8, dropout=0., stage_steps=5, edge_increment=.5)
    return dict(smoke=True, family='expected_graph_selftraining',
        implementation_module='expected_graph_selftraining_v1',
        source_sha256={'expected_graph_selftraining_v1.py': experiment.common.base.digest(core.__file__)},
        core_parameters=parameters, kinds=list(core.KINDS), epochs=10, check_interval=5,
        weight_decay=.001, strong_weight=.1)


class ExpectedGraphExecutorTests(unittest.TestCase):
    def test_full_and_observed_control_have_identical_warmup(self):
        a, _ = fixture()
        a['anchor_oof'] = np.zeros_like(a['y_train'], dtype=np.float64)
        full, _, first, _ = experiment.fit(a, protocol(), 'full', .001, 29, 'cpu', fixed_epochs=5)
        null, _, second, _ = experiment.fit(a, protocol(), 'observed_only', .001, 29, 'cpu', fixed_epochs=5)
        for key in full.state_dict():
            torch.testing.assert_close(full.state_dict()[key], null.state_dict()[key], atol=0, rtol=0)
        self.assertEqual(first, second)

    def test_fixed_training_and_setup_reject_queries_and_other_gpu(self):
        a, _ = fixture()
        a['anchor_oof'] = np.zeros_like(a['y_train'], dtype=np.float64)
        contaminated = dict(a, y_val=np.ones(3))
        with self.assertRaisesRegex(ValueError, 'training inputs only'):
            experiment.fit(contaminated, protocol(), 'full', .001, 1, 'cpu', fixed_epochs=1)
        with self.assertRaisesRegex(ValueError, 'training inputs only'):
            experiment.setup(contaminated, protocol(), 'full', 'cpu')
        with self.assertRaisesRegex(ValueError, 'cuda:1'):
            experiment.fit(a, protocol(), 'full', .001, 1, 'cuda:0', fixed_epochs=1)

    def test_all_arms_cross_an_actual_teacher_boundary_and_identity_does_not_train(self):
        a, _ = fixture()
        a['anchor_oof'] = np.zeros_like(a['y_train'], dtype=np.float64)
        p = protocol()
        for kind in core.KINDS:
            model, value, trace, detail = experiment.fit(a, p, kind, .001, 17, 'cpu', fixed_epochs=10)
            self.assertIsNone(value)
            self.assertEqual(detail['gradient_evaluations'], 10)
            self.assertEqual(detail['graph_forward_evaluations'], 20)
            self.assertEqual(detail['teacher_refreshes'], 2)
            self.assertEqual(len(trace), 10)
            self.assertEqual([e['before_step'] for e in model.augmentation_evidence], [1, 6])
            self.assertEqual([row['stage'] for row in trace], [0] * 5 + [1] * 5)
            self.assertEqual(any(row['virtual_edges'] for row in trace), kind != 'observed_only')
            for stage in model.augmentation_evidence:
                for evidence in stage['contexts']:
                    self.assertFalse(set(map(tuple, evidence['added_pairs'].T)) &
                                     set(map(tuple, evidence['query_pairs'].T)))
            ln = dict(a, pair_offset=np.array([0, 14]))
            model, _, trace, detail = experiment.fit(ln, p, kind, .001, 17, 'cpu', fixed_epochs=0)
            self.assertEqual(model.state_dict(), {})
            self.assertEqual(model.augmentation_evidence, [])
            self.assertFalse(detail['trained_neural_model'])
            self.assertEqual(detail['training_steps'], 0)
            self.assertEqual(trace, [])


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
