import unittest
import pandas as pd

from improve.biopath_experiment import assessment


class PathGateTests(unittest.TestCase):
    def rows(self):
        values={'MiDrug':{'parent':(.95,.91),'full':(.951,.912),'no_brg':(.9505,.911),'shuffled_brg':(.9504,.9109)},
                'LncDrug':{'parent':(.988,.975),'full':(.988,.975),'no_brg':(.988,.975),'shuffled_brg':(.988,.975)}}
        return [dict(task=task,kind=kind,gnn_seed=seed,auc=v[0],ap=v[1],macro_drug_ap=.7)
                for task,arms in values.items() for kind,v in arms.items() for seed in (666,0,1)]

    def test_large_lnc_ap_loss_cannot_pass(self):
        rows=self.rows()
        for row in rows:
            if row['task']=='LncDrug' and row['kind']=='full':row['ap']-=.002
        result=assessment(pd.DataFrame(rows))
        self.assertTrue(result['mechanism_supported']);self.assertFalse(result['shared_gate'])

    def test_no_mechanism_win_cannot_pass(self):
        rows=self.rows()
        for row in rows:
            if row['task']=='MiDrug' and row['kind']=='shuffled_brg':row['ap']=.913
        result=assessment(pd.DataFrame(rows))
        self.assertTrue(result['performance']);self.assertFalse(result['shared_gate'])


if __name__=='__main__':unittest.main()
