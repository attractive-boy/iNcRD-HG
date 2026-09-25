"""The confirmation must not hide split failures behind an average."""
import unittest

import pandas as pd

from improve.confirm_reification_kernel import assess


class ConfirmationGateTests(unittest.TestCase):
    def frame(self):
        rows=[]
        for task in ('MiDrug','LncDrug'):
            for split in (42,2026,31415):
                for kind,gain in [('pretrained',.003),('kernel',.001),('random',.0015)]:
                    for score,delta in [('gnn',0.),('gnn_composed',gain)]:
                        rows.append(dict(task=task,split=split,kind=kind,score=score,
                                         auc=.95+delta,ap=.9+delta,macro_drug_ap=.8+delta))
        return pd.DataFrame(rows)

    def test_every_lnc_split_must_respect_tolerance(self):
        frame=self.frame()
        self.assertTrue(assess(frame)['shared_gate'])
        hit=(frame.task=='LncDrug')&(frame.kind=='pretrained')&(frame.score=='gnn_composed')&(frame.split==42)
        frame.loc[hit,'ap']=.898
        result=assess(frame)
        self.assertGreater(result['lnc_vs_gnn']['delta_ap'],0.)
        self.assertFalse(result['lnc_shared_performance'])
        self.assertFalse(result['shared_gate'])

    def test_one_large_win_cannot_pass_mechanism_sign_gate(self):
        frame=self.frame()
        hit=(frame.task=='MiDrug')&(frame.kind=='pretrained')&(frame.score=='gnn_composed')
        frame.loc[hit,'ap']=[.920,.901,.901]
        result=assess(frame)
        self.assertGreater(result['mi_pretrained_vs_controls']['random']['delta_ap'],0.)
        self.assertFalse(result['mechanism_supported'])

    def test_missing_partition_cannot_be_silently_averaged(self):
        frame=self.frame()
        frame=frame[~((frame.kind=='random')&(frame.split==42))]
        with self.assertRaises(ValueError):assess(frame)


if __name__=='__main__':unittest.main()
