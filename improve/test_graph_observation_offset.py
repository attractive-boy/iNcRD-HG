import unittest

import numpy as np

from improve.graph_observation_offset import likelihood, fit_effects, context_offsets


class ObservationOffsetTest(unittest.TestCase):
    def test_analytic_likelihood_gradient(self):
        pairs = np.array([[0, 0, 1, 1, 2], [0, 1, 0, 1, 0]])
        y = np.array([1., 0., 1., 1., 0.])
        x = np.random.default_rng(63).normal(size=6)
        _, actual = likelihood(x, pairs, y, (3, 2), .7)
        expected = []
        for j in range(len(x)):
            delta = np.zeros_like(x)
            delta[j] = 1e-5
            expected.append((likelihood(x + delta, pairs, y, (3, 2), .7)[0] -
                             likelihood(x - delta, pairs, y, (3, 2), .7)[0]) / 2e-5)
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)

    def test_balanced_complete_matrix_has_zero_main_effects(self):
        pairs = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
        r, d, c, details = fit_effects(pairs, np.array([1, 0, 0, 1]), (2, 2))
        np.testing.assert_array_equal(r, np.zeros(2))
        np.testing.assert_array_equal(d, np.zeros(2))
        self.assertEqual(c, 0.)
        self.assertTrue(details['optimizer_success'])

    def test_heldout_fold_label_does_not_change_its_support_offset(self):
        pairs = np.array([[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]])
        a = dict(node_counts=np.array([3, 2]), pair_offset=np.array([0, 3]), nd=np.array(2),
                 pairs_train=pairs, y_train=np.array([1, 0, 0, 1, 1, 0]), folds=np.array([0, 0, 1, 1, 1, 1]))
        first = context_offsets(a, folds=2)
        changed = dict(a, y_train=a['y_train'].copy())
        changed['y_train'][:2] = 1 - changed['y_train'][:2]
        second = context_offsets(changed, folds=2)
        np.testing.assert_array_equal(first[0][0], second[0][0])
        self.assertEqual(first[1][0], second[1][0])

    def test_input_order_does_not_change_convex_fit(self):
        pairs = np.array([[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]])
        y = np.array([1, 0, 1, 1, 0, 0])
        first = fit_effects(pairs, y, (3, 2))
        second = fit_effects(pairs[:, ::-1], y[::-1], (3, 2))
        for a, b in zip(first[:3], second[:3]):
            np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
