import unittest
import numpy as np
from improve.rna_functional_modules import project, stratified_permutation, Representations, VARIANTS


class RepresentationTests(unittest.TestCase):
    def test_query_attributes_cannot_change_fitted_projection(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(40,25))
        support = np.arange(25)
        q,a,_ = project(x,np.ones(40,bool),support)
        changed = x.copy()
        changed[25:] *= 100
        r,b,_ = project(changed,np.ones(40,bool),support)
        for key in a:
            np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(q[:25],r[:25])

    def test_missing_vectors_zero_and_support_entities_not_frequency_weighted(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(30,20))
        present = np.ones(30,bool)
        present[-2:] = False
        a,state,_ = project(x,present,np.repeat(np.arange(20),np.arange(1,21)))
        b,_,_ = project(x,present,np.arange(20))
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[-2:],0)
        self.assertEqual(len(state['fit_ids']),20)

    def test_stratified_derangement_preserves_coverage_support_and_metadata(self):
        n = 80
        simple = np.ones((n,8),np.float32)
        simple[:,2] = np.repeat([1,2],40)
        simple[:,3] = np.tile(np.repeat([.4,.6],20),2)
        present = np.ones(n,bool)
        present[-5:] = False
        support = np.arange(60)
        p,meta = stratified_permutation(simple,present,support)
        np.testing.assert_array_equal(p,stratified_permutation(simple,present,support)[0])
        np.testing.assert_array_equal(p[-5:],np.arange(75,80))
        np.testing.assert_array_equal(simple[p],simple)
        self.assertEqual(set(p),set(range(n)))
        self.assertTrue(set(p[support])<=set(support))
        self.assertGreater(meta['moved_fraction'],.9)

    def test_all_ablations_have_same_width_and_simple_metadata(self):
        rng = np.random.default_rng(1)
        n = 40
        data = dict(simple=rng.uniform(size=(n,8)),present=np.ones(n,bool),names=np.arange(n),
                    generic=rng.normal(size=(n,64)),functional_mean=rng.normal(size=(n,30)),
                    functional_moments=rng.normal(size=(n,60)))
        tables = Representations({'task':data})
        for variant in VARIANTS:
            values,_,_ = tables.transform('task',variant,np.arange(25))
            self.assertEqual(values.shape,(n,24))
            np.testing.assert_array_equal(values[:,:8],data['simple'])


if __name__=='__main__':
    unittest.main()
