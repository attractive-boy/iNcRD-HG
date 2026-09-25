"""Training boundaries, graph-mean KL gradients and exact identity deployment."""
import unittest

import numpy as np
import torch

from improve import anchor_bottleneck_experiment_v1 as experiment
from improve import information_bottleneck_graph_v1 as core
from improve.test_nested_anchor_v2 import fixture


def protocol():
    parameters = core.parameters()
    parameters.update(width=8, dropout=0.)
    return dict(smoke=True, family='information_bottleneck',
        implementation_module='information_bottleneck_graph_v1',
        source_sha256={'information_bottleneck_graph_v1.py': experiment.common.base.digest(core.__file__)},
        core_parameters=parameters, kinds=list(core.KINDS), epochs=2, check_interval=1,
        weight_decay=.001, strong_weight=.1, private_rng_offset=29015)


class BottleneckExecutorTests(unittest.TestCase):
    def test_fixed_training_and_setup_reject_evaluation_arrays_and_other_gpu(self):
        a, _ = fixture()
        a['anchor_oof'] = np.zeros_like(a['y_train'], dtype=np.float64)
        contaminated = dict(a, y_val=np.ones(3))
        with self.assertRaisesRegex(ValueError, 'training inputs only'):
            experiment.fit(contaminated, protocol(), 'full', .001, 1, 'cpu', fixed_epochs=1)
        with self.assertRaisesRegex(ValueError, 'training inputs only'):
            experiment.setup(contaminated, protocol(), 'full', 'cpu')
        with self.assertRaisesRegex(ValueError, 'cuda:1'):
            experiment.fit(a, protocol(), 'full', .001, 1, 'cuda:0', fixed_epochs=1)

    def test_objective_averages_per_query_kl_and_no_kl_removes_its_gradient(self):
        p = protocol()
        for kind in core.KINDS:
            residual = torch.tensor([.2, -.5], dtype=torch.float64, requires_grad=True)
            kl = torch.tensor([3., 11.], dtype=torch.float64, requires_grad=True)
            loss, bce, mean_kl, weighted = experiment.objective(torch.zeros(2), residual,
                torch.tensor([1., 0.]), kl, kind, p)
            self.assertEqual(float(mean_kl), 7.)
            self.assertAlmostEqual(float(loss - bce), 0. if kind == 'no_kl' else .007, places=12)
            loss.backward()
            if kind == 'no_kl':
                self.assertIsNone(kl.grad)
                self.assertEqual(float(weighted), 0.)
            else:
                torch.testing.assert_close(kl.grad, torch.full_like(kl, .0005), atol=0, rtol=0)
            self.assertTrue(torch.isfinite(residual.grad).all())

    def test_cpu_short_fits_and_lnc_identity(self):
        a, _ = fixture()
        a['anchor_oof'] = np.zeros_like(a['y_train'], dtype=np.float64)
        p = protocol()
        for kind in core.KINDS:
            model, value, trace, detail = experiment.fit(a, p, kind, .001, 17, 'cpu', fixed_epochs=2)
            self.assertIsNone(value)
            self.assertTrue(detail['trained_neural_model'])
            self.assertEqual(detail['gradient_evaluations'], 2)
            self.assertEqual(len(trace), 2)
            self.assertEqual(detail['kl_contributes_to_gradient'], kind != 'no_kl')
            self.assertEqual(sum(row['weighted_kl'] for row in trace) == 0., kind == 'no_kl')
            ln = dict(a, pair_offset=np.array([0, 14]))
            model, _, trace, detail = experiment.fit(ln, p, kind, .001, 17, 'cpu', fixed_epochs=0)
            self.assertEqual(model.state_dict(), {})
            self.assertEqual(trace, [])
            self.assertFalse(detail['trained_neural_model'])
            self.assertEqual(detail['training_steps'], 0)
            _, xs, graphs, _ = experiment.setup(ln, p, kind, 'cpu')
            self.assertEqual(graphs, [None] * 5)
            scores = experiment.predict(model, xs, graphs, torch.tensor([[0, 1], [14, 15]]), p)
            torch.testing.assert_close(scores, torch.zeros(2, dtype=torch.float64), atol=0, rtol=0)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
