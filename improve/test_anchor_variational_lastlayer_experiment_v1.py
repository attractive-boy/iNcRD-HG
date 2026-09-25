"""Exercise posterior training, aligned inference and the identity boundary."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from improve import anchor_variational_lastlayer_experiment_v1 as experiment
from improve import variational_lastlayer_graph_v1 as core
from improve.test_nested_anchor_v2 import fixture


def protocol():
    parameters = core.parameters()
    parameters.update(width=8, heads=2)
    return dict(smoke=True, family='variational_lastlayer', implementation_module='variational_lastlayer_graph_v1',
        source_sha256={'variational_lastlayer_graph_v1.py': experiment.common.base.digest(core.__file__)},
        core_parameters=parameters, kinds=list(core.KINDS), epochs=5, check_interval=20,
        weight_decay=.001, strong_weight=.1)


class VariationalExecutorTests(unittest.TestCase):
    def test_fixed_training_boundary_and_covariance_gradients(self):
        a, _ = fixture()
        a['anchor_oof'] = np.linspace(-2, 2, len(a['y_train']))
        p = protocol()
        with self.assertRaisesRegex(ValueError, 'training arrays only'):
            experiment.fit(dict(a, y_val=np.ones(3)), p, 'full', .001, 17, 'cpu', fixed_epochs=1)
        with self.assertRaisesRegex(ValueError, 'cuda:1'):
            experiment.fit(a, p, 'full', .001, 17, 'cuda:0', fixed_epochs=1)
        for kind in core.KINDS:
            model, value, trace, detail = experiment.fit(a, p, kind, .001, 17, 'cpu', fixed_epochs=5)
            self.assertIsNone(value)
            self.assertEqual(detail['training_steps'], 5)
            self.assertEqual(detail['gradient_evaluations'], 5)
            self.assertTrue(all('anchor_composed' not in row for row in trace))
            self.assertTrue(all(np.isfinite(row['loss']) for row in trace))
            self.assertEqual(detail['regularizer_is_gaussian_kl'], kind != 'map')
            self.assertGreater(float(model.last.mean.norm()), 0)
            if kind == 'full':
                self.assertGreater(float(model.last.lower.norm()), 0)
            else:
                self.assertEqual(float(model.last.lower.norm()), 0)

    def test_saved_posterior_replays_with_aligned_anchor_and_quadrature(self):
        a, _ = fixture()
        a['anchor_oof'] = np.linspace(-2, 2, len(a['y_train']))
        p = protocol()
        model, _, _, _ = experiment.fit(a, p, 'full', .001, 17, 'cpu', fixed_epochs=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save(model.state_dict(), path)
            loaded, xs, graphs, _ = experiment.setup(a, p, 'full', 'cpu')
            loaded.load_state_dict(torch.load(path, weights_only=True))
            pairs = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None])
            anchor = torch.linspace(-3, 3, pairs.shape[1], dtype=torch.float64)
            actual = experiment.predict(loaded, xs, graphs, pairs, p, anchor)
            reverse = experiment.predict(loaded, xs, graphs, pairs.flip(1), p, anchor.flip(0)).flip(0)
            batched = torch.cat([experiment.predict(loaded, xs, graphs, pairs[:, i:i+3], p, anchor[i:i+3])
                                 for i in range(0, pairs.shape[1], 3)])
            reference = experiment.predict(loaded, xs, graphs, pairs, p, anchor, 128)
            for other in (reverse, batched, reference):
                torch.testing.assert_close(actual, other, atol=3e-5, rtol=3e-5)
            with self.assertRaisesRegex(ValueError, 'aligned'):
                experiment.predict(loaded, xs, graphs, pairs, p, anchor[:-1])

    def test_identity_does_not_train_or_construct_posterior(self):
        a, _ = fixture()
        a['anchor_oof'] = np.zeros_like(a['y_train'], dtype=np.float64)
        a['pair_offset'] = np.array([0, 14])
        p = protocol()
        for kind in core.KINDS:
            model, _, trace, detail = experiment.fit(a, p, kind, .001, 17, 'cpu', fixed_epochs=0)
            self.assertFalse(detail['trained_neural_model'])
            self.assertEqual(detail['parameters'], 0)
            self.assertEqual(trace, [])
            loaded, xs, graphs, _ = experiment.setup(a, p, kind, 'cpu')
            pairs = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None])
            anchor = torch.linspace(-2, 2, pairs.shape[1], dtype=torch.float64)
            components = experiment.predict_components(loaded, xs, graphs, pairs, p, anchor)
            for value in components.values():
                torch.testing.assert_close(value, torch.zeros_like(value), atol=0, rtol=0)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
