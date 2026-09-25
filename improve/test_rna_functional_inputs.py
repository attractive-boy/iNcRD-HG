"""Scientific input invariants for the matched RNA representation experiment."""
import unittest
import numpy as np
from improve.rna_functional_inputs import moments, requested_accession, resolve_accession


class InputTests(unittest.TestCase):
    def test_weighted_local_variance_and_missingness(self):
        local = np.array([[[1., 2.], [3., 4.]], [[9., 9.], [20., 20.]]], np.float32)
        mean, sd = moments(local, [[1., 3.], [0., 0.]])
        np.testing.assert_allclose(mean, [[2.5, 3.5], [0., 0.]])
        np.testing.assert_allclose(sd, [[np.sqrt(.75)]*2, [0., 0.]])

    def test_versioned_refseq_must_not_silently_change(self):
        self.assertTrue(requested_accession('NM_005485.4', ['NM_005485.4']))
        self.assertTrue(requested_accession('NM_005485.4', ['NM_005485']))
        self.assertFalse(requested_accession('NM_005485.5', ['NM_005485.4']))
        self.assertIsNone(resolve_accession('NM_005485.4', {'NM_005485.5':'ACGT'}))
        self.assertEqual(resolve_accession('NM_005485', {'NM_005485.4':'ACGT'}), 'NM_005485.4')
        with self.assertRaises(ValueError):
            resolve_accession('NM_005485', {'NM_005485.4':'A', 'NM_005485.5':'C'})


if __name__ == '__main__':
    unittest.main()
