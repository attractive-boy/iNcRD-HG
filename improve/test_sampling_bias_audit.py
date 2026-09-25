import unittest
import numpy as np
from improve.sampling_bias_audit import matched_weights, entity_features, weighted_metrics


class SamplingAuditTests(unittest.TestCase):
    def test_matching_removes_entity_only_discrimination_and_preserves_prevalence(self):
        y=np.array([1,1,1,0,1,0,0,0,1])
        group=np.array([0,0,0,0,1,1,1,1,2])
        w,info=matched_weights(y,group)
        self.assertEqual(info['retained_pairs'],8)
        self.assertEqual(w[-1],0)
        self.assertAlmostEqual(np.average(y,weights=w),y[w>0].mean())
        self.assertAlmostEqual(weighted_metrics(y,group,w)['auc'],.5)
        for entity in [0,1]:
            self.assertAlmostEqual(w[(group==entity)&(y==1)].sum()/w[y==1].sum(),
                                   w[(group==entity)&(y==0)].sum()/w[y==0].sum())

    def test_query_order_and_missing_entities(self):
        support=np.array([[0,0,1,1],[0,1,0,1]])
        labels=np.array([1,0,0,1])
        q=np.array([[2,0,1],[2,0,1]])
        x,_=entity_features(support,labels,q,(3,3))
        xp,_=entity_features(support,labels,q[:,::-1],(3,3))
        np.testing.assert_array_equal(x,xp[::-1])
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(x[0,0],0)
        self.assertEqual(x[0,3],0)

    def test_no_eligible_stratum(self):
        w,info=matched_weights(np.array([0,0,1]),np.array([0,0,1]))
        self.assertEqual(info['retained_pairs'],0)
        self.assertIsNone(weighted_metrics(np.array([0,0,1]),np.arange(3),w)['auc'])


if __name__=='__main__':
    unittest.main()
