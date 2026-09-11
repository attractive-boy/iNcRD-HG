"""Meaningful checks for optimizer routing, scaling and robust inner selection."""
import copy
import unittest

import numpy as np
import pandas as pd
import torch

from improve.dual_pair_modules import DualPairModel
from improve.dual_pair_tuning import (TunedDualPairModel, choose_epoch, configuration_grid,
                                     make_optimizer, setting)
from improve.test_pair_state import ToyBase, toy_graph


class DualPairTuningTests(unittest.TestCase):
    def test_default_matches_existing_architecture_and_scaled_gradient(self):
        graph,pairs=toy_graph(),torch.tensor([[0,1,2],[2,0,1]])
        base=ToyBase()
        torch.manual_seed(19)
        original=DualPairModel(copy.deepcopy(base),graph,6,'lncRNA','dual_pair',width=4)
        torch.manual_seed(19)
        model=TunedDualPairModel(copy.deepcopy(base),graph,6,'lncRNA','dual_pair',setting('x',1.),width=4)
        for k,v in original.state_dict().items():
            torch.testing.assert_close(model.state_dict()[k],v,atol=0,rtol=0)
        with torch.no_grad():
            model.branch.head[-1].weight.normal_(std=.1)
        original.load_state_dict(model.state_dict())
        model.correction_scale=.3
        a=original(graph.x_dict,graph.edge_index_dict,pairs,{},True)
        b=model(graph.x_dict,graph.edge_index_dict,pairs,{},True)
        torch.testing.assert_close(b[2],.3*a[2])
        torch.testing.assert_close(b[0],b[1]+b[2])
        g1=torch.autograd.grad(a[2].sum(),original.branch.head[-1].weight)[0]
        g2=torch.autograd.grad(b[2].sum(),model.branch.head[-1].weight)[0]
        torch.testing.assert_close(g2,.3*g1)

    def test_optimizer_groups_cover_parameters_once_at_intended_rates(self):
        graph=toy_graph()
        cfg=setting('x',.3,1e-4,10.,.3,.5)
        model=TunedDualPairModel(ToyBase(),graph,6,'lncRNA','dual_pair',cfg,width=4)
        optimizer=make_optimizer(model,.001,cfg)
        ids=[id(p) for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters()})
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'],.0003)
        self.assertAlmostEqual(optimizer.param_groups[1]['lr'],.003)
        self.assertTrue(all(group['weight_decay']==1e-4 for group in optimizer.param_groups))
        for block in list(model.branch.blocks)+list(model.branch.local_blocks):
            for value in block.step.values():
                self.assertAlmostEqual(float(value.sigmoid()),.5)

    def test_scaled_matched_controls_have_equal_parameters_and_transport_identity(self):
        graph,pairs=toy_graph(),torch.tensor([[0,1,2],[2,0,1]])
        base,cfg=ToyBase(),setting('x',.3,1e-4,3.,.3,.1)
        models=[]
        for arm in ('dual_pair','matched_local','matched_transport'):
            torch.manual_seed(29)
            models.append(TunedDualPairModel(copy.deepcopy(base),graph,6,'lncRNA',arm,cfg,width=4))
        with torch.no_grad():
            models[0].branch.head[-1].weight.normal_(std=.1)
        for model in models[1:]:
            model.load_state_dict(models[0].state_dict())
        for model in models:
            actual=model(graph.x_dict,graph.edge_index_dict,pairs,{},intervention='no_transport')
            expected=models[1](graph.x_dict,graph.edge_index_dict,pairs,{})
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        self.assertEqual(len({sum(p.numel() for p in m.parameters()) for m in models}),1)

    def test_multiseed_smoothing_prefers_shared_stable_epoch_and_rejects_outer(self):
        def trace(values):
            return pd.DataFrame(dict(epoch=np.arange(1,len(values)+1),inner_auc=values,
                                     inner_aupr=values,criterion=values))
        a,b=trace([.95,.3,.7,.8,.8]),trace([.1,.3,.7,.8,.8])
        selected=choose_epoch([a,b],2)
        self.assertEqual(selected['epochs'],5)
        self.assertAlmostEqual(selected['selection_score'],.8)
        self.assertEqual(choose_epoch([trace([.5,.5,.5])],2)['epochs'],2)
        with self.assertRaises(ValueError):
            choose_epoch([pd.DataFrame(dict(epoch=[1,2],loss=[1.,.5],queries=[4,4]))],2)
        with self.assertRaises(ValueError):
            choose_epoch([a,b.iloc[:-1]],2)
        grids=configuration_grid()
        self.assertEqual(len(grids['pca_query']),len(grids['dual_pair']))
        self.assertEqual(len(grids['dual_pair']),12)


if __name__=='__main__':
    unittest.main()
