import io
import unittest

import numpy as np
import torch

from improve import graph_tabm
from improve.numeric_embedding import ARMS, model_for
from improve.numeric_embedding_modules import (
    EmbeddedTabM, PiecewiseLinearEncoding, fit_bin_edges,
)


class NumericalEmbeddingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_piecewise_values_boundaries_and_clamped_tails(self):
        basis = PiecewiseLinearEncoding([np.array([0., 2., 5.])], n_bins=3)
        x = torch.tensor([[-3.], [0.], [1.], [2.], [3.5], [5.], [10.]])
        expected = torch.tensor([[0., 0., 0.], [0., 0., 0.], [.5, 0., 0.],
                                 [1., 0., 0.], [1., .5, 0.], [1., 1., 0.], [1., 1., 0.]])
        torch.testing.assert_close(basis(x)[:, 0], expected)

    def test_duplicate_quantiles_and_constants_are_finite(self):
        x = np.array([[0., 4.]] * 30 + [[1., 4.]] * 10 + [[2., 4.]] * 8, dtype=np.float32)
        edges = fit_bin_edges(x, n_bins=16)
        self.assertTrue(all(np.all(np.diff(e) > 0) for e in edges))
        basis = PiecewiseLinearEncoding(edges, n_bins=16)
        values = basis(torch.from_numpy(x))
        self.assertTrue(torch.isfinite(values).all())
        self.assertEqual(int(torch.count_nonzero(values[:, 1])), 0)
        self.assertEqual(int(torch.count_nonzero(values * (1 - basis.valid))), 0)

    def test_tree_recovers_a_supervised_threshold_from_fitting_rows(self):
        x = np.linspace(-1, 1, 101, dtype=np.float32)[:, None]
        y = (x[:, 0] > .17).astype(np.float32)
        edges = fit_bin_edges(x, y, n_bins=8, method='tree', min_leaf=5)[0]
        self.assertEqual(len(edges), 3)
        self.assertAlmostEqual(float(edges[1]), .17, places=6)
        np.testing.assert_array_equal(edges[[0, -1]], [-1., 1.])

    def test_quantile_knots_ignore_labels(self):
        x = np.random.RandomState(2).normal(size=(80, 3)).astype(np.float32)
        a = fit_bin_edges(x, np.zeros(len(x)), n_bins=8, method='quantile')
        b = fit_bin_edges(x, np.ones(len(x)), n_bins=8, method='quantile')
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)

    def test_inference_does_not_refit_knots_on_evaluation_rows(self):
        x = np.arange(48, dtype=np.float32).reshape(16, 3)
        basis = PiecewiseLinearEncoding(fit_bin_edges(x, n_bins=8), n_bins=8)
        original = {k: v.clone() for k, v in basis.state_dict().items()}
        basis(torch.full((20, 3), 1e9))
        basis(torch.full((30, 3), -1e9))
        for k, v in basis.state_dict().items():
            torch.testing.assert_close(v, original[k])

    def test_invalid_knots_and_labels_are_rejected(self):
        for edge in ([0., 0., 1.], [2., 1.], [0., float('nan')]):
            with self.assertRaises(ValueError):
                PiecewiseLinearEncoding([np.array(edge)], n_bins=4)
        with self.assertRaises(ValueError):
            fit_bin_edges(np.ones((10, 3)), np.ones(9), method='tree')

    def test_embeddings_have_gradients_and_batch_order_invariance(self):
        torch.manual_seed(3)
        x = torch.randn(24, 4)
        y = torch.tensor([0., 1., 0.] * 8)
        for edges in (None, fit_bin_edges(x.numpy(), n_bins=8)):
            model = EmbeddedTabM(4, embedding_dim=6, edges=edges, n_bins=8,
                                 width=16, depth=2, members=4, dropout=0.)
            output = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output, y[:, None].expand_as(output))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertGreater(float(model.embedding.weight.grad.abs().sum()), 0.)
            self.assertTrue((model.backbone.output_weight.grad.abs().sum(1) > 0).all())
            model.eval()
            torch.testing.assert_close(model(x), torch.cat([model(x[:7]), model(x[7:])]))
            order = torch.randperm(len(x))
            torch.testing.assert_close(model(x)[order], model(x[order]))

    def test_checkpoint_preserves_knots_and_predictions(self):
        x = np.random.RandomState(9).normal(size=(40, 3)).astype(np.float32)
        model = EmbeddedTabM(3, embedding_dim=4, edges=fit_bin_edges(x, n_bins=8),
                             n_bins=8, width=8, depth=2, members=3, dropout=0.)
        model.eval()
        expected = model(torch.from_numpy(x))
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        # A freshly constructed model with different knots must replay exactly.
        restored = EmbeddedTabM(3, embedding_dim=4, edges=fit_bin_edges(x * 2, n_bins=8),
                                n_bins=8, width=8, depth=2, members=3, dropout=0.)
        restored.load_state_dict(torch.load(stream, weights_only=False))
        restored.eval()
        torch.testing.assert_close(restored(torch.from_numpy(x)), expected)

    def test_scalar_control_is_exactly_existing_tabm(self):
        cfg = dict(width=16, depth=2, members=4, dropout=.1, embedding_dim=4, n_bins=8, min_leaf=4)
        x = np.random.RandomState(4).normal(size=(30, 4)).astype(np.float32)
        y = np.array([0., 1., 0.] * 10, dtype=np.float32)
        original = graph_tabm.model_for('tabm', 4, cfg, 666, 'cpu')
        control = model_for('tabm', 4, cfg, 666, 'cpu', x, y)
        for k, v in original.state_dict().items():
            torch.testing.assert_close(control.state_dict()[k], v, atol=0, rtol=0)
        for arm in ARMS:
            output = model_for(arm, 4, cfg, 666, 'cpu', x, y)(torch.from_numpy(x))
            self.assertEqual(output.shape, (30, 4))
            self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
