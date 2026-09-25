import unittest
import numpy as np
import pandas as pd
import torch

from improve.functional_response_transfer import masked_mse,normalization,within_lineage_permutation,assay_metrics,assess_metrics


class FunctionalAssayTests(unittest.TestCase):
    def test_unobserved_assays_have_zero_loss_and_gradient(self):
        p=torch.tensor([[1.,1e6]],requires_grad=True)
        y=torch.tensor([[0.,-1e6]])
        value=masked_mse(p,y,torch.tensor([[True,False]]));value.backward()
        self.assertEqual(value.item(),1.)
        torch.testing.assert_close(p.grad,torch.tensor([[2.,0.]]),atol=0,rtol=0)

    def test_held_out_phenotype_and_expression_do_not_fit_scalers(self):
        a=dict(x=np.arange(30,dtype=np.float32).reshape(10,3),
               gene=np.arange(20,dtype=np.float32).reshape(10,2),
               drug=np.arange(20,dtype=np.float32).reshape(10,2))
        original=normalization(a,np.arange(8))
        for name in ('x','gene','drug'):a[name][8:]=1e20
        changed=normalization(a,np.arange(8))
        for name in original:np.testing.assert_array_equal(original[name],changed[name])

    def test_shuffle_preserves_lineages_and_is_a_permutation(self):
        groups=np.array(['a','a','a','b','b','c'])
        order=within_lineage_permutation(groups,31)
        np.testing.assert_array_equal(np.sort(order),np.arange(6))
        np.testing.assert_array_equal(groups,groups[order])
        self.assertTrue(np.all(order[:5]!=np.arange(5)))
        self.assertEqual(order[5],5)

    def test_missing_assays_do_not_change_reported_errors(self):
        truth=np.array([[1.,np.nan],[3.,2.]],dtype=np.float32)
        prediction=np.array([[0.,1e6],[1.,0.]],dtype=np.float32)
        stats=(np.zeros(2),np.ones(2),np.ones(2,dtype=bool))
        first=assay_metrics(truth,prediction,*stats)
        prediction[0,1]=-1e6
        second=assay_metrics(truth,prediction,*stats)
        self.assertEqual(first,second)
        self.assertAlmostEqual(first['normalized_mse'],3.)
        self.assertEqual(first['observations'],3)

    def test_joint_must_beat_control_even_if_it_beats_mean(self):
        rows=[]
        for kind,error in [('joint',.8),('drug_only',.7),('random_encoder',.95),('cell_shuffled',.96)]:
            for fold in range(5):
                for assay in ('gene','drug','matched_drug'):
                    rows.append(dict(kind=kind,fold=fold,assay=assay,normalized_mse=error,training_mean_baseline_mse=1.))
        p=dict(fold_count=5,gate=dict(mean_mse_advantage=0.,min_positive_folds=3,
               baseline_assays=['gene','drug','matched_drug'],
               control_assays={'drug':['drug_only','random_encoder','cell_shuffled'],
                               'gene':['random_encoder','cell_shuffled']}))
        passed,checks=assess_metrics(pd.DataFrame(rows),p)
        self.assertFalse(passed)
        self.assertTrue(all(r['passed'] for r in checks if r['control']=='training_mean'))
        self.assertFalse(next(r['passed'] for r in checks if r['control']=='drug_only'))


if __name__=='__main__':unittest.main()
