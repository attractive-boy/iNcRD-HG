"""Training-only contract, actual gradient counts, and zero-fit identity records."""
import unittest

import numpy as np

from improve import anchor_sharpness_experiment_v1 as experiment
from improve import sharpness_graph_v1 as core
from improve.test_nested_anchor_v2 import fixture


def parameters():
    p = core.parameters(); p.update(width=8, heads=2, dropout=0.)
    return dict(smoke=True, family='friendly_sharpness', implementation_module='sharpness_graph_v1',
        source_sha256={'sharpness_graph_v1.py': experiment.common.base.digest(core.__file__)},
        core_parameters=p, kinds=list(core.KINDS), epochs=5, check_interval=5,
        weight_decay=.001, strong_weight=.1, private_rng_offset=29015,
        optimizer_parameters=dict(rho=.05, sigma=1., decay=.9, clip=5.))


class SharpnessExecutorTests(unittest.TestCase):
    def test_all_midrug_arms_have_two_gradients_per_actual_training_step(self):
        a, _ = fixture()
        a['anchor_oof'] = np.linspace(-1., 1., len(a['y_train']))
        p = parameters()
        for kind in core.KINDS:
            model, values, trace, detail = experiment.fit(a, p, kind, .003, 426, 'cpu', fixed_epochs=5)
            self.assertIsNone(values)
            self.assertTrue(detail['trained_neural_model'])
            self.assertFalse(detail['identity_preserves_anchor'])
            self.assertEqual(detail['training_steps'], 5)
            self.assertEqual(detail['gradient_evaluations'], 10)
            self.assertEqual(len(trace), 1)
            self.assertNotIn('anchor_composed', trace[0])
            self.assertGreater(model.readout[-1].weight.abs().sum().item(), 0.)
        with self.assertRaisesRegex(ValueError, 'training inputs only'):
            experiment.fit(dict(a, y_val=np.ones(7)), p, 'full', .003, 426, 'cpu', fixed_epochs=5)

    def test_lnc_records_do_not_fit_or_read_query_labels(self):
        a, _ = fixture()
        keep = a['pairs_train'][0] < int(a['node_counts'][0])
        a['pairs_train'] = a['pairs_train'][:, keep]
        for key in ('y_train', 'folds'):
            a[key] = a[key][keep]
        a['pair_offset'] = np.array([0, 14])
        a['anchor_oof'] = np.linspace(-1., 1., len(a['y_train']))
        p = parameters()
        for kind in core.KINDS:
            model, value, trace, detail = experiment.fit(a, p, kind, .003, 426, 'cpu', fixed_epochs=0)
            self.assertIsNone(value)
            self.assertEqual(model.state_dict(), {})
            self.assertEqual(trace, [])
            for name in ('parameters', 'trainable_parameters', 'training_steps', 'gradient_evaluations', 'epoch'):
                self.assertEqual(detail[name], 0)
            self.assertFalse(detail['trained_neural_model'])
            self.assertTrue(detail['identity_preserves_anchor'])
            # Only query identity is supplied. An identity path must not require
            # query outcomes or learn a model even in the historical interface.
            arrays = dict(a, pairs_val=a['pairs_train'][:, :5])
            _, value, _, _ = experiment.fit(arrays, p, kind, .003, 426, 'cpu')
            np.testing.assert_array_equal(value, np.zeros(5))


if __name__ == '__main__':
    unittest.main()
