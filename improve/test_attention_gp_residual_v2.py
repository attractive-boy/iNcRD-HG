"""Convex MAP equivalence and complete target-label isolation."""
import unittest

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch

from improve import attention_gp_residual_v2 as core


class ResidualGPTests(unittest.TestCase):
    def test_map_matches_independent_primal_logistic_solution(self):
        rng = np.random.default_rng(20812)
        x = rng.normal(size=(31, 7))
        y = rng.binomial(1, .4, len(x)).astype(float)
        offset = rng.normal(size=len(x))
        def objective(beta):
            logits = offset + x @ beta
            value = np.logaddexp(0., logits).sum() - y @ logits + 5.*(beta @ beta)
            gradient = x.T @ (expit(logits)-y) + 10.*beta
            return value, gradient
        oracle = minimize(objective, np.zeros(7), jac=True, method='BFGS', options=dict(gtol=1e-10))
        alpha, trace = core.solve(torch.from_numpy(x @ x.T), torch.from_numpy(y), torch.from_numpy(offset))
        np.testing.assert_allclose(x @ oracle.x, x @ x.T @ alpha.numpy(), atol=2e-7, rtol=2e-7)
        self.assertLessEqual(trace[-1]['maximum_dual_residual'], 1e-7)
        self.assertTrue(all(b['objective'] <= a['objective']+1e-9 for a,b in zip(trace,trace[1:])))

    def test_universe_prediction_and_permutation(self):
        rng = np.random.default_rng(30912)
        xr, xd = rng.normal(size=(6, 4)), rng.normal(size=(5, 3))
        kr, kd = torch.from_numpy(xr @ xr.T), torch.from_numpy(xd @ xd.T)
        pairs = np.array([[0, 1, 2, 2, 3, 5], [1, 3, 2, 0, 4, 2]])
        y = np.array([0., 1., 1., 0., 1., 0.])
        offset = rng.normal(size=6)
        first, _, _, error = core.fit(kr, kd, pairs, y, offset)
        order = rng.permutation(6)
        second, _, _, _ = core.fit(kr, kd, pairs[:,order], y[order], offset[order])
        np.testing.assert_allclose(first, second, atol=1e-8, rtol=1e-8)
        self.assertLess(error, 1e-8)

    def test_all_target_edges_removed_and_labels_unused(self):
        rng = np.random.default_rng(41211)
        a = dict(pairs_train=np.array([[0, 1], [0, 1]]), y_train=np.array([1, 0]),
                 nd=np.array(2), folds=np.array([0, 1]), node_counts=np.array([2, 2, 2, 2]),
                 pair_offset=np.array([2, 4]), edges=np.array([[2, 3, 0, 1, 6], [4, 5, 4, 3, 2]]),
                 relation=np.array([0, 0, 1, 2, 3]), target_codes=np.array([0, 3, -1, -1, -1]),
                 **{'x_'+str(i): rng.normal(size=(2, 3)) for i in range(4)})
        aux = core.auxiliary_arrays(a)
        self.assertTrue(np.all(aux['target_codes'] < 0))
        for kind in core.KINDS:
            first = core.entity_kernels(a, kind)
            second = core.entity_kernels(dict(a, y_train=1-a['y_train']), kind)
            for left, right in zip(first, second):
                np.testing.assert_array_equal(left, right)
        with self.assertRaises(ValueError):
            core.auxiliary_arrays(dict(a, y_val=np.ones(2)))


if __name__ == '__main__':
    unittest.main()
