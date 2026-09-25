"""Outcome isolation and binary-flow mechanics used in the experiment."""
import unittest

import numpy as np
import torch

from improve import discrete_flow_models as method


class FlowTest(unittest.TestCase):
    def test_oof_labels_do_not_change_support_prior_or_loss_targets(self):
        pairs = np.array([[0, 1, 0, 2], [0, 1, 1, 0]])
        y = np.array([1, 1, 0, 1])
        folds = np.array([0, 1, 0, 1])
        query = np.array([[2], [1]])
        a, mask = method.support_and_mask(pairs, y, folds, 0, (3, 2), query)
        other = y.copy()
        other[folds == 0] = 1 - other[folds == 0]
        b, mask_b = method.support_and_mask(pairs, other, folds, 0, (3, 2), query)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(mask, mask_b)
        self.assertFalse(mask[0].any())
        self.assertFalse(mask[2, 1])

    def test_binary_path_and_continuous_expectation(self):
        torch.manual_seed(91)
        target = torch.tensor([[1., 0., 0., 1.], [0., 1., 0., 0.]])
        prior = torch.rand_like(target)
        mask = torch.rand_like(target)
        at_zero = method.interpolated_state(target, prior, mask, 0., 'discrete')
        at_one = method.interpolated_state(target, prior, mask, 1., 'discrete')
        torch.testing.assert_close(at_one, target)
        continuous = method.interpolated_state(target, prior, mask, .3, 'continuous')
        torch.testing.assert_close(continuous, .3 * target + .7 * at_zero)
        binary = method.interpolated_state(target, prior, mask, .3, 'discrete')
        self.assertTrue(bool(((binary == 0) | (binary == 1)).all()))

    def test_euler_preserves_observed_and_binary_support(self):
        known = torch.tensor([[1., 0., 0.]])
        state = known.clone()
        prediction = torch.tensor([[-10., 2., -.3]])
        actual = method.binary_euler(state, prediction, known, .8, .1)
        torch.testing.assert_close(actual, torch.tensor([[1., 1., 0.]]))

    def test_query_order_and_batch_do_not_change_scores(self):
        torch.manual_seed(92)
        model = method.BinaryFlowModel(5, 3, width=8, time_dim=4).eval()
        graph = (torch.rand(7, 5) > .8).float()
        attrs = torch.randn(7, 3)
        full = method.predict(model, graph, attrs, 'discrete', batch=7)
        reverse = method.predict(model, graph, attrs, 'discrete',
                                 rows=np.arange(6, -1, -1), batch=2)
        np.testing.assert_allclose(reverse[::-1], full, atol=2e-6, rtol=2e-6)


if __name__ == '__main__':
    unittest.main()
