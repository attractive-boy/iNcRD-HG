import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from improve.context_functional_eval import source_mask, bootstrap_groups, bootstrap_delta, design, auxiliary_pmids


class FunctionalSourceTests(unittest.TestCase):
    def test_any_shared_source_and_missing_source_excluded(self):
        mask,counts = source_mask([{'pmids':'11;12'},{'pmids':'13'},{'pmids':''}],{'12'})
        np.testing.assert_array_equal(mask,[False,True,False])
        self.assertEqual(counts['retained_sites'],1)
        self.assertEqual(counts['missing_source'],1)

    def test_integer_pmid_csv_serialization_is_normalized_exactly(self):
        self.assertEqual(auxiliary_pmids('23622248.0;12'),['12','23622248'])
        mask,_ = source_mask([{'pmids':'23622248.0'}],{'23622248'})
        self.assertFalse(mask[0])
        with self.assertRaises(ValueError):
            auxiliary_pmids('23622248.5')

    def test_bootstrap_connects_multi_source_triples(self):
        codes = bootstrap_groups(pd.DataFrame({'source_groups':['["a", "b"]','["b"]','["c"]']}))
        self.assertEqual(codes[0],codes[1])
        self.assertNotEqual(codes[1],codes[2])

    def test_paired_bootstrap_identical_predictors_have_zero_delta(self):
        y = np.array([0,1,0,1,0,1])
        score = np.array([.2,.8,.4,.6,.3,.7])
        result = bootstrap_delta(y,score,score,np.repeat(np.arange(3),2),repeats=10)
        self.assertEqual(result['valid_resamples'],10)
        self.assertEqual(result['intervals'],{'auc':[0.,0.],'ap':[0.,0.]})

    def test_numeric_context_interaction_has_fixed_column_budget(self):
        identity = sparse.csr_matrix([[1,0],[0,1]])
        contexts = sparse.csr_matrix([[1,0,1],[0,1,1]])
        numeric = np.array([[2.,3.],[4.,5.]])
        matrix = design(identity,contexts,numeric).toarray()
        self.assertEqual(matrix.shape,(2,10))
        np.testing.assert_array_equal(matrix[:,4:7],[[2,0,2],[0,4,4]])


if __name__=='__main__':
    unittest.main()
