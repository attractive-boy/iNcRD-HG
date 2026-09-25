import tempfile
from pathlib import Path
import unittest

import numpy as np

from improve.drug_multimodal_features import DrugFeatures,drug_permutation,VARIANTS


class DrugFeatureTests(unittest.TestCase):
    def test_shuffling_retains_support_coverage_and_component_count(self):
        n = 80
        simple = np.ones((n,8))
        simple[:,7] = np.repeat([0.,1.],40)
        present = np.ones(n,bool)
        present[-3:] = False
        support = np.arange(60)
        p,meta = drug_permutation(simple,present,support)
        np.testing.assert_array_equal(p,drug_permutation(simple,present,support)[0])
        np.testing.assert_array_equal(present[p],present)
        np.testing.assert_array_equal(simple[p,7],simple[:,7])
        self.assertTrue(set(p[support])<=set(support))
        self.assertEqual(set(p),set(range(n)))
        self.assertGreater(meta['moved_fraction'],.9)

    def test_same_dimensions_missing_fallback_and_query_boundary(self):
        rng = np.random.default_rng(12)
        n = 30
        simple = rng.normal(size=(n,8))
        simple[:,7] = 0
        present = np.ones(n,bool)
        present[-2:] = False
        table = dict(simple=simple,present=present,names=np.arange(n),cids=np.arange(n))
        for name,width in [('m2umol',128),('chemberta',768),('fingerprint',2048)]:
            table[name] = rng.normal(size=(n,width))
            table[name][~present] = 0
        with tempfile.TemporaryDirectory() as directory:
            np.savez(Path(directory)/'representations.npz',**table)
            enc = DrugFeatures(directory)
            support = np.stack([np.arange(20),np.arange(20)])
            for variant in VARIANTS:
                x,_,_ = enc.transform(variant,support)
                self.assertEqual(x.shape,(n,24))
                np.testing.assert_array_equal(x[:,:8],simple)
                np.testing.assert_array_equal(x[~present,8:],0)
            a,state,_ = enc.transform('m2umol',support)
            enc.table['m2umol'][20:28] *= 100
            b,state2,_ = enc.transform('m2umol',support)
            for key in state:
                np.testing.assert_array_equal(state[key],state2[key])
            np.testing.assert_array_equal(a[:20],b[:20])


if __name__=='__main__':
    unittest.main()
