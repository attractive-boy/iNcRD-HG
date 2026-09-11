"""Numerical ranking checks, including positive-only threshold ties."""
from itertools import product
import unittest

import numpy as np

from improve.analyze_pair_state import numerical_metric_envelope
from improve.network_feature_ablation import score


class MetricEnvelopeTests(unittest.TestCase):
    def assert_covered(self, y, reference, delta):
        low, high = numerical_metric_envelope(y, reference, float(np.abs(delta).max()))
        observed = np.array(score(y, reference+delta))
        self.assertTrue(np.all(observed >= low-1e-12), (observed,low))
        self.assertTrue(np.all(observed <= high+1e-12), (observed,high))

    def test_exhaustive_small_interval_perturbations(self):
        y = np.array([1,0,1,1,0])
        reference = np.array([3.,2.,1.,1.,1.000001])
        for delta in product((-1e-6,0.,1e-6),repeat=5):
            self.assert_covered(y,reference,np.array(delta))

    def test_breaking_positive_only_tie_changes_ap(self):
        y = np.array([1,0,1,1])
        reference = np.array([3.,2.,1.,1.])
        delta = np.array([0.,0.,0.,1e-6])
        self.assertNotEqual(score(y,reference)[1],score(y,reference+delta)[1])
        self.assert_covered(y,reference,delta)

    def test_separated_scores_have_exact_metric_bounds(self):
        y = np.array([1,0,1,0])
        reference = np.array([4.,3.,2.,1.])
        low,high=numerical_metric_envelope(y,reference,1e-6)
        np.testing.assert_allclose(low,score(y,reference),atol=1e-12,rtol=0)
        np.testing.assert_allclose(high,low,atol=1e-12,rtol=0)


if __name__ == '__main__':
    unittest.main()
