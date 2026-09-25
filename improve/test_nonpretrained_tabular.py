"""Test evidence controls, fresh fitting, and ranking-safe checkpoint replay."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from improve import nonpretrained_tabular_compare as experiment
from improve import nonpretrained_tabular_modules as models


CFG=dict(width=16,depth=2,members=4,dropout=0.,lr=.001,weight_decay=.0001)


def arrays():
    rng=np.random.default_rng(19)
    a={}
    for part,n in (('train',80),('val',20)):
        a['raw_'+part]=rng.normal(size=(n,36)).astype(np.float32)
        a['y_'+part]=np.tile([0,1],n//2).astype(np.float32)
        for view in ('tiered62','expanded62'):
            a[view+'_raw_extra_'+part]=rng.normal(size=(n,26)).astype(np.float32)
    return a


class InputControls(unittest.TestCase):
    def test_tree_columns_keep_exact_raw_evidence(self):
        a=arrays()
        for view in ('graph36','tiered62','expanded62'):
            x,q=models.tree_inputs(a,view)
            for value,part,n in ((x,'train',80),(q,'val',20)):
                self.assertEqual(value.shape,(n,62))
                np.testing.assert_array_equal(value[:,:36],a['raw_'+part])
                expected=(np.zeros((n,26)) if view=='graph36' else a[view+'_raw_extra_'+part])
                np.testing.assert_array_equal(value[:,36:],expected)

    def test_queries_do_not_change_training_inputs(self):
        a=arrays();first=models.tree_inputs(a,'expanded62')[0]
        a['raw_val']*=1000;a['expanded62_raw_extra_val']+=100
        a['y_val']=1-a['y_val']
        np.testing.assert_array_equal(models.tree_inputs(a,'expanded62')[0],first)

    def test_invalid_views_and_outer_access_rejected(self):
        with self.assertRaises(ValueError):
            models.tree_inputs(arrays(),'unknown')
        with patch.object(experiment.previous.np,'load',side_effect=AssertionError('Outer array opened')) as read:
            with self.assertRaisesRegex(ValueError,'inner data only'):
                experiment.load_inputs('/unused',dict(tasks=['MiDrug']),'MiDrug','outer')
            read.assert_not_called()


class TabMControls(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_stable_scores_preserve_mean_probability_aggregation(self):
        logits=torch.tensor([[-2.,1.,3.],[4.,-3.,.2]],dtype=torch.float64)
        expected=logits.sigmoid().mean(1)
        score=models.ensemble_log_odds(logits)
        torch.testing.assert_close(score.sigmoid(),expected,atol=1e-12,rtol=1e-12)

    def test_extreme_member_scores_do_not_collapse_to_ties(self):
        logits=torch.tensor([[100.,101.],[110.,111.],[-100.,-101.],[-110.,-111.]])
        score=models.ensemble_log_odds(logits).numpy()
        self.assertTrue(np.isfinite(score).all())
        self.assertGreater(score[1],score[0])
        self.assertLess(score[3],score[2])

    def test_initialization_and_capacity_match_across_views(self):
        states=set();counts=set()
        for _ in ('graph36','tiered62','expanded62'):
            model=models.make_tabm(CFG,666,'cpu')
            states.add(models.state_digest(model))
            counts.add(sum(p.numel() for p in model.parameters()))
        self.assertEqual(len(states),1);self.assertEqual(len(counts),1)
        self.assertNotEqual(models.state_digest(models.make_tabm(CFG,0,'cpu')),
                            models.state_digest(models.make_tabm(CFG,1,'cpu')))

    def test_full_support_gradient_accumulation_and_added_columns(self):
        a=arrays();x,_=models.tree_inputs(a,'expanded62')
        x=torch.from_numpy(x);y=torch.from_numpy(a['y_train'])
        fitted=[];losses=[]
        for batch in (80,17):
            model=models.make_tabm(CFG,0,'cpu')
            optimizer=torch.optim.SGD(model.parameters(),lr=.02)
            before=models.state_digest(model)
            losses.append(models.train_tabm_epoch(model,optimizer,x,y,batch))
            self.assertNotEqual(before,models.state_digest(model))
            self.assertTrue(bool(torch.isfinite(model.blocks[0].weight.grad).all()))
            self.assertGreater(float(model.blocks[0].weight.grad[:,36:].abs().sum()),0.)
            fitted.append(model)
        self.assertAlmostEqual(losses[0],losses[1],places=6)
        for key,tensor in fitted[0].state_dict().items():
            torch.testing.assert_close(tensor,fitted[1].state_dict()[key],atol=1e-6,rtol=1e-6)

    def test_replay_and_query_batch_invariance(self):
        a=arrays();x,q=models.tree_inputs(a,'tiered62')
        model=models.make_tabm(CFG,1,'cpu')
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        models.train_tabm_epoch(model,optimizer,torch.from_numpy(x),torch.from_numpy(a['y_train']),17)
        q=torch.from_numpy(q);score=models.predict_tabm(model,q,20)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt';torch.save(model.state_dict(),path)
            rng=torch.get_rng_state().clone()
            with torch.random.fork_rng(devices=[]):
                restored=models.make_tabm(CFG,1,'cpu')
                restored.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
                np.testing.assert_allclose(models.predict_tabm(restored,q,7),score,atol=1e-6,rtol=1e-6)
                np.testing.assert_allclose(models.predict_tabm(restored,q.flip(0),6)[::-1],score,atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(rng,torch.get_rng_state())


class TreeControls(unittest.TestCase):
    def test_fit_api_excludes_external_initialization_and_validation(self):
        class Fixture:
            def is_fitted(self):
                return False
            def fit(self,*args,**kwargs):
                self.args=args;self.kwargs=kwargs;return self
        a=arrays();x,_=models.tree_inputs(a,'graph36');model=Fixture()
        models.fit_catboost(model,x,a['y_train'])
        self.assertEqual(len(model.args),2)
        self.assertEqual(model.kwargs,dict(verbose=False))

    def test_fresh_tree_prefix_replay_and_query_invariance(self):
        dependency=Path(experiment.DEPENDENCIES)/'packages'
        cfg=dict(iterations=12,learning_rate=.1,depth=3,loss_function='Logloss',
                 task_type='CPU',thread_count=2,allow_writing_files=False,use_best_model=False,verbose=False)
        a=arrays();x,q=models.tree_inputs(a,'expanded62')
        model=models.fresh_catboost(cfg,666,dependency)
        self.assertFalse(model.is_fitted())
        models.fit_catboost(model,x,a['y_train'])
        self.assertEqual(set(model.get_evals_result()),{'learn'})
        with self.assertRaisesRegex(ValueError,'fitted'):
            models.fit_catboost(model,x,a['y_train'])
        prefix=model.copy();prefix.shrink(ntree_end=5)
        score=models.predict_catboost(prefix,q)
        np.testing.assert_allclose(score,model.predict(q,prediction_type='RawFormulaVal',ntree_end=5),atol=1e-12,rtol=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.cbm';prefix.save_model(str(path))
            restored=models.catboost_class(dependency)();restored.load_model(str(path))
            self.assertEqual(restored.tree_count_,5)
            np.testing.assert_array_equal(models.predict_catboost(restored,q),score)
            np.testing.assert_array_equal(models.predict_catboost(restored,q[::-1])[::-1],score)
            np.testing.assert_array_equal(np.concatenate([models.predict_catboost(restored,p) for p in np.array_split(q,3)]),score)


if __name__=='__main__':
    unittest.main()
