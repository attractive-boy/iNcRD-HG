import unittest

import numpy as np

from improve.graph_mechanism_validate import endpoint_intervals,train_fixed


class ValidationTest(unittest.TestCase):
    def test_query_outcomes_cannot_enter_training(self):
        with self.assertRaisesRegex(ValueError,'Fresh query data'):
            train_fixed({'y_val':np.array([0,1])},{},'full',{},0,'cpu','block')

    def test_paired_identical_scores_have_zero_interval(self):
        y=np.tile([0,1],6)
        pairs=np.array([np.repeat(np.arange(6),2),np.tile([0,1,2],4)])
        pred=np.linspace(0,1,len(y))
        cell=dict(y=y,pairs=pairs,scores={'full':pred,'anchor':pred.copy()})
        interval=endpoint_intervals([cell],repeats=40,seed=712)
        for metric in ('auc','ap'):
            for end in ('low','high'):
                self.assertEqual(interval['intervals']['anchor'][metric][end],0.)

    def test_duplicating_partition_does_not_narrow_cluster_interval(self):
        y=np.tile([0,1],6)
        pairs=np.array([np.repeat(np.arange(6),2),np.tile([0,1,2],4)])
        cell=dict(y=y,pairs=pairs,scores={'full':np.arange(12),'anchor':np.arange(12)[::-1]})
        single=endpoint_intervals([cell],repeats=60,seed=17)
        duplicated=endpoint_intervals([cell,cell],repeats=60,seed=17)
        for metric in ('auc','ap'):
            for end in ('low','high','positive_fraction'):
                self.assertAlmostEqual(single['intervals']['anchor'][metric][end],
                                       duplicated['intervals']['anchor'][metric][end],places=12)


if __name__=='__main__':
    unittest.main()
