"""Independent derivative and score-invariance checks for scalar calibration."""
import unittest

import numpy as np
from scipy.optimize import approx_fprime

from improve import parametric_cut_calibration_v1 as core


class CalibrationTests(unittest.TestCase):
    def test_derivative_and_stationary_solution(self):
        rng = np.random.default_rng(80431)
        rank = rng.uniform(0., 1., 120)
        offset = rng.normal(0., .5, len(rank))
        y = rng.binomial(1, 1 / (1 + np.exp(-(offset + 1.5 * (rank - .5)))))
        x = (rank - rank.mean()) / rank.std()
        theta = np.array([.31, -.2])
        actual = core.objective(theta, x, y, offset, .01)[1]
        expected = approx_fprime(theta, lambda z: core.objective(z, x, y, offset, .01)[0], 1e-7)
        np.testing.assert_allclose(actual, expected, atol=3e-8, rtol=1e-6)
        fitted = core.fit(rank, y, offset)
        self.assertLess(fitted['gradient_max_abs'], 1e-7)
        self.assertLessEqual(fitted['objective'], core.objective(np.zeros(2), x, y, offset, .01)[0])

    def test_constant_rank_and_query_invariance(self):
        y = np.tile([0., 1.], 30)
        fitted = core.fit(np.ones(len(y)), y, np.zeros(len(y)))
        self.assertEqual(fitted['beta'], 0.)
        np.testing.assert_array_equal(core.predict(np.ones(19), fitted), np.zeros(19))
        fitted.update(beta=.7, intercept=-.3, mean=.4, scale=.2)
        query = np.linspace(-.3, 1.7, 29)
        score = core.predict(query, fitted)
        np.testing.assert_array_equal(score, core.predict(query[::-1], fitted)[::-1])
        np.testing.assert_array_equal(score, np.concatenate([core.predict(z, fitted) for z in np.array_split(query, 4)]))
        self.assertLessEqual(abs(score).max(), 2.)


if __name__ == '__main__':
    unittest.main()
