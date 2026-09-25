"""Training-only inputs, proximal stage completion, and identity isolation."""
import unittest

import numpy as np
import torch

from improve import anchor_ranking_experiment_v1 as experiment
from improve import staco_ranking_v1 as core
from improve.test_nested_anchor_v2 import fixture


def protocol():
    p = dict(core.parameters(), width=8, depth=1, dropout=0., inner_steps=2,
             positive_batch=2, negative_batch=2)
    return dict(smoke=True, family='staco_ranking', implementation_module='staco_ranking_v1',
        source_sha256={'staco_ranking_v1.py': experiment.common.base.digest(core.__file__)},
        core_parameters=p, kinds=list(core.KINDS), epochs=4, check_interval=2,
        weight_decay=.001, strong_weight=.1, private_rng_offset=29015)


class RankingExecutorChecks(unittest.TestCase):
    def test_fixed_fit_rejects_evaluation_arrays_and_other_cuda_devices(self):
        a, _ = fixture()
        a['anchor_oof'] = np.zeros_like(a['y_train'], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, 'training inputs only'):
            experiment.fit(dict(a, y_val=np.ones(3)), protocol(), 'full', .001, 1, 'cpu', fixed_epochs=4)
        with self.assertRaisesRegex(ValueError, 'cuda:1'):
            experiment.fit(a, protocol(), 'full', .001, 1, 'cuda:0', fixed_epochs=4)

    def test_all_controls_complete_two_stages_and_identity_skips_optimizer(self):
        a, _ = fixture()
        a['anchor_oof'] = np.zeros_like(a['y_train'], dtype=np.float64)
        for kind in core.KINDS:
            model, value, trace, detail = experiment.fit(a, protocol(), kind, .001, 1, 'cpu', fixed_epochs=4)
            self.assertIsNone(value)
            self.assertTrue(detail['trained_neural_model'])
            self.assertEqual(detail['gradient_evaluations'], 4)
            self.assertEqual(detail['completed_proximal_stages'], 2)
            self.assertTrue(all(t['stage_averaged'] for t in trace))
            self.assertEqual(model.training_risk_snapshot['selected_training_step'], 4)
            self.assertEqual(model.training_risk_snapshot['updates'], 4)
            self.assertTrue(torch.isfinite(model.training_risk_snapshot['s']).all())
        ln = dict(a, pair_offset=np.array([0, 14]))
        model, value, trace, detail = experiment.fit(ln, protocol(), 'full', .001, 1, 'cpu', fixed_epochs=0)
        self.assertFalse(detail['trained_neural_model'])
        self.assertEqual(model.state_dict(), {})
        self.assertEqual(detail['training_steps'], 0)
        self.assertEqual(trace, [])
        self.assertEqual(model.training_risk_snapshot, dict(identity=True, training_steps=0))


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
