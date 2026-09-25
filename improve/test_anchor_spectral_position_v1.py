"""Formal device restrictions, training-only inputs, and exact LncDrug identity."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from improve import anchor_priority_graph_v1 as experiment
from improve import stable_spectral_position_v1 as core
from improve.test_nested_anchor_v2 import fixture


def protocol():
    p = dict(core.parameters(), width=8, depth=1, dropout=0., pe_width=4, pe_filters=3, psi_width=5)
    return dict(smoke=True, family='stable_spectral_position', implementation_module='stable_spectral_position_v1',
        source_sha256={'stable_spectral_position_v1.py': experiment.common.base.digest(core.__file__)},
        core_parameters=p, kinds=list(core.KINDS), epochs=2, check_interval=1,
        weight_decay=.001, strong_weight=.1, private_rng_offset=29015)


class SpectralExecutorChecks(unittest.TestCase):
    def test_fixed_fit_rejects_evaluation_arrays_and_other_cuda_devices(self):
        a, _ = fixture()
        a['anchor_oof'] = np.zeros_like(a['y_train'], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, 'training inputs only'):
            experiment.fit(dict(a, y_val=np.ones(3)), protocol(), 'full', .001, 1, 'cpu', fixed_epochs=2)
        with self.assertRaisesRegex(ValueError, 'cuda:1'):
            experiment.fit(a, protocol(), 'full', .001, 1, 'cuda:0', fixed_epochs=2)

    def test_controls_train_and_identity_does_not_decompose_any_graph(self):
        a, _ = fixture()
        a['anchor_oof'] = np.zeros_like(a['y_train'], dtype=np.float64)
        for kind in core.KINDS:
            model, value, trace, detail = experiment.fit(a, protocol(), kind, .001, 1, 'cpu', fixed_epochs=2)
            self.assertIsNone(value)
            self.assertTrue(detail['trained_neural_model'])
            self.assertEqual(detail['gradient_evaluations'], 2)
            self.assertEqual(len(trace), 2)
        ln = dict(a, pair_offset=np.array([0, 14]))
        with patch.object(core, 'eigensystem', side_effect=AssertionError('Identity cannot build spectrum')):
            model, _, trace, detail = experiment.fit(ln, protocol(), 'full', .001, 1, 'cpu', fixed_epochs=0)
        self.assertFalse(detail['trained_neural_model'])
        self.assertEqual(model.state_dict(), {})
        self.assertEqual(detail['training_steps'], 0)
        self.assertEqual(trace, [])


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
