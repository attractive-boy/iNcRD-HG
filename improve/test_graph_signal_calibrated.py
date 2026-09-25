"""Cross-fitting has to remove the query before graph/spectrum construction."""
import unittest

import numpy as np

from improve.graph_signal_calibrated import (
    apply_calibration, fit_calibration, make_folds, support_matrix,
)


class CrossFittingTests(unittest.TestCase):
    def test_excluded_positive_never_enters_graph(self):
        pairs = np.array([[0,0,1,1,2,2],[0,1,0,1,0,1]])
        y = np.array([1,0,1,0,1,0])
        folds = make_folds(pairs, y, 11, count=3)
        for fold in range(3):
            a = support_matrix(pairs, y, folds, fold, (3,2))
            excluded = pairs[:,folds==fold]
            self.assertFalse(a[excluded[0],excluded[1]].any())
            changed_y = y.copy()
            changed_y[folds==fold] = 1-changed_y[folds==fold]
            b = support_matrix(pairs, changed_y, folds, fold, (3,2))
            np.testing.assert_array_equal(a,b)

    def test_duplicate_pair_rejected_before_cross_fitting(self):
        with self.assertRaises(ValueError):
            make_folds(np.array([[0,0,0,1],[0,0,1,1]]), np.array([1,1,0,0]), 1, count=2)

    def test_saved_calibration_matches_and_ignores_evaluation_batch(self):
        scores = np.array([-.4,.7,.2,1.8,2.2,-1.3])
        y = np.array([0,1,0,1,1,0])
        c = fit_calibration(scores,y)
        query = np.array([-1.,0.,.5,2.])
        full = apply_calibration(query,c)
        np.testing.assert_array_equal(full,apply_calibration(query[::-1],c)[::-1])
        np.testing.assert_array_equal(full,np.concatenate([apply_calibration(query[:2],c),apply_calibration(query[2:],c)]))
        self.assertGreater(c['coefficient'],0)
        self.assertEqual(c['rows'],len(y))


if __name__ == '__main__':
    unittest.main()
