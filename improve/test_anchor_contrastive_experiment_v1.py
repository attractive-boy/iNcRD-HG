"""No evaluation inputs in fixed training, and no fitting on the identity arm."""
import unittest

import numpy as np
import torch

from improve import anchor_contrastive_experiment_v1 as experiment
from improve import neighbor_type_contrastive_v1 as cooperative
from improve.test_nested_anchor_v2 import fixture


def protocol():
    core = cooperative.parameters()
    core.update(width=8, depth=2, dropout=0.)
    return dict(smoke=True, family='neighbor_type_contrastive', implementation_module='neighbor_type_contrastive_v1',
        source_sha256={'neighbor_type_contrastive_v1.py': experiment.common.base.digest(cooperative.__file__)},
        core_parameters=core, kinds=list(cooperative.KINDS), epochs=2, check_interval=1,
        weight_decay=.001, strong_weight=.1, private_rng_offset=29015)


class PriorityExecutorTests(unittest.TestCase):
    def test_fixed_training_rejects_queries_and_other_cuda_devices(self):
        a, _ = fixture(); a['anchor_oof'] = np.zeros_like(a['y_train'], dtype=np.float64)
        contaminated = dict(a, y_val=np.ones(3))
        with self.assertRaisesRegex(ValueError, 'training inputs only'):
            experiment.fit(contaminated, protocol(), 'full', .001, 1, 'cpu', fixed_epochs=1)
        with self.assertRaisesRegex(ValueError, 'cuda:1'):
            experiment.fit(a, protocol(), 'full', .001, 1, 'cuda:0', fixed_epochs=1)

    def test_cpu_smoke_trains_mid_only_and_identity_has_empty_state(self):
        a, _ = fixture(); a['anchor_oof'] = np.zeros_like(a['y_train'], dtype=np.float64)
        p = protocol()
        model, value, trace, detail = experiment.fit(a, p, 'full', .001, 1, 'cpu', fixed_epochs=2)
        self.assertTrue(detail['trained_neural_model'])
        self.assertEqual(detail['gradient_evaluations'], 2)
        self.assertIsNone(value)
        ln = dict(a, pair_offset=np.array([0, 14]))
        model, value, trace, detail = experiment.fit(ln, p, 'full', .001, 1, 'cpu', fixed_epochs=0)
        self.assertFalse(detail['trained_neural_model'])
        self.assertEqual(model.state_dict(), {})
        self.assertEqual(detail['training_steps'], 0)
        self.assertEqual(trace, [])
        _, xs, graphs, _ = experiment.setup(ln, p, 'full', 'cpu')
        query = torch.tensor([[0, 1, 2], [14, 15, 16]])
        actual = experiment.predict(model, xs, graphs, query, p)
        torch.testing.assert_close(actual, torch.zeros(3, dtype=torch.float64), atol=0, rtol=0)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
