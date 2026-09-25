import unittest

import numpy as np

from improve.frontier_report_guard import diagnostic_matched_weights
from improve.sampling_bias_audit import matched_weights,weighted_metrics


class DiagnosticReportTests(unittest.TestCase):
    def test_empty_or_one_class_subset_is_undefined_not_an_error_or_perfect_score(self):
        for labels in ([],[0,0,0],[1,1]):
            y = np.asarray(labels,int)
            w,_ = diagnostic_matched_weights(y,np.arange(len(y)))
            np.testing.assert_array_equal(w,np.zeros(len(y)))
            result = weighted_metrics(y,np.ones(len(y))*.3,w)
            self.assertIsNone(result['auc'])
            self.assertIsNone(result['ap'])

    def test_all_two_class_matching_results_remain_identical(self):
        y = np.asarray([1,0,0,1,0,1])
        groups = np.asarray([1,1,1,2,3,3])
        a,meta = diagnostic_matched_weights(y,groups)
        b,original = matched_weights(y,groups)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(meta,original)


if __name__=='__main__':
    unittest.main()
