"""Controls that make the evidence-addition comparison interpretable."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from improve import gnn_tabicl_compare as compare
from improve import gnn_tabicl_compare_modules as models
from improve.pair_exchange_modules import restrict_training_graph
from improve.test_ragt_backbone import graph_fixture


CFG=dict(width=16,depth=1,heads=4,dropout=.1,fusion='add',gate_init=-2.)


class FeatureControls(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(17)
        self.a=dict(x_train=rng.normal(size=(40,36)).astype(np.float32),
                    x_val=rng.normal(size=(12,36)).astype(np.float32),
                    y_train=np.tile([0,1],20),y_val=np.tile([0,1],6))
        self.extra=[rng.normal(size=(n,26)).astype(np.float32) for n in (40,12,40,12)]

    def test_query_changes_do_not_change_support_normalization(self):
        train,query=self.extra[:2]
        x,q,quantiles,refs=compare.normalized_extra(train,query)
        other=compare.normalized_extra(train,query*1000+2000)
        np.testing.assert_array_equal(x,other[0])
        np.testing.assert_array_equal(quantiles,other[2])
        np.testing.assert_array_equal(refs,other[3])
        self.assertFalse(np.array_equal(q,other[1]))

    def test_feature_budget_and_frozen_graph_columns(self):
        conditions=compare.feature_conditions(self.a,*self.extra)
        for view in compare.VIEWS:
            for part,n in (('train',40),('val',12)):
                x=conditions[view+'_'+part]
                self.assertEqual(x.shape,(n,62))
                np.testing.assert_array_equal(x[:,:36],self.a['x_'+part])
                self.assertTrue(np.isfinite(x).all())
                if view=='graph36':
                    np.testing.assert_array_equal(x[:,36:],np.zeros((n,26)))
        np.testing.assert_array_equal(conditions['tiered62_raw_extra_train'],self.extra[0])
        np.testing.assert_array_equal(conditions['expanded62_raw_extra_val'],self.extra[3])
        self.assertFalse(np.array_equal(conditions['tiered62_train'][:,36:],
                                       conditions['expanded62_train'][:,36:]))

    def test_invalid_feature_budget_fails(self):
        with self.assertRaisesRegex(ValueError,'26'):
            compare.feature_conditions(self.a,self.extra[0][:,:25],*self.extra[1:])
        bad=dict(self.a,x_train=self.a['x_train'][:,:35])
        with self.assertRaisesRegex(ValueError,'36'):
            compare.feature_conditions(bad,*self.extra)

    def test_outer_rejected_before_any_array_load(self):
        with patch.object(compare.np,'load',side_effect=AssertionError('Unexpected array access')) as read:
            for phase in ('outer','test'):
                with self.assertRaisesRegex(ValueError,'inner data only'):
                    compare.load_features('/nonexistent',dict(tasks=['LncDrug']),'LncDrug',phase)
            read.assert_not_called()


class NeuralControls(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.g,self.pairs,self.labels=graph_fixture()
        self.weights={key:torch.ones(edge.shape[1]) for key,edge in self.g.edge_index_dict.items()}

    def make(self,architecture,seed=666):
        return models.make_model(architecture,CFG,self.g.x_dict,self.g.edge_index_dict,
                                 'lncRNA',seed,'cpu')

    def test_same_head_and_full_initialization_across_input_conditions(self):
        head_digests=set()
        for architecture in compare.ARCHITECTURES:
            digests=set();counts=set()
            for _ in compare.VIEWS:
                torch.manual_seed(1982)
                model=self.make(architecture)
                digests.add(models.state_digest(model))
                counts.add(sum(p.numel() for p in model.parameters()))
                head_digests.add(models.state_digest(model.feature_mlp))
            self.assertEqual(len(digests),1)
            self.assertEqual(len(counts),1)
        self.assertEqual(len(head_digests),1)
        self.assertNotEqual(models.state_digest(self.make('mlp',0)),
                            models.state_digest(self.make('mlp',1)))

    def test_sampled_queries_and_holdout_edges_hidden_in_both_directions(self):
        support_pairs,support_y=self.pairs[:,:-3],self.labels[:-3]
        g,weights=restrict_training_graph(self.g,self.weights,'LncDrug','lncRNA',
                                          support_pairs,support_y)
        support_positive=set(map(tuple,support_pairs[:,support_y==1].T))
        for epoch in (1,2):
            idx,visible=models.training_context(support_pairs,support_y,g.edge_index_dict,weights,
                'LncDrug','lncRNA',g['drug'].num_nodes,.5,666,epoch)
            repeat,_=models.training_context(support_pairs,support_y,g.edge_index_dict,weights,
                'LncDrug','lncRNA',g['drug'].num_nodes,.5,666,epoch)
            np.testing.assert_array_equal(idx,repeat)
            self.assertEqual(set(support_y[idx]),{0,1})
            held=set(map(tuple,support_pairs[:,idx[support_y[idx]==1]].T))
            for key,edges in visible.items():
                if key[1] in ('LncDrug','rev_LncDrug'):
                    oriented=edges if key[0]=='lncRNA' else edges.flip(0)
                    self.assertEqual(set(map(tuple,oriented.numpy().T)),support_positive-held)
                else:
                    torch.testing.assert_close(edges,g[key].edge_index)
            self.assertEqual(set(map(tuple,g['lncRNA','LncDrug','drug'].edge_index.numpy().T)),
                              support_positive)

    def test_added_evidence_affects_predictions_and_receives_finite_gradients(self):
        features=torch.tensor(np.random.default_rng(12).normal(size=(len(self.labels),62)),
                              dtype=torch.float32)
        baseline=features.clone();baseline[:,36:]=0
        pairs=torch.from_numpy(self.pairs)
        for architecture in compare.ARCHITECTURES:
            model=self.make(architecture)
            first=models.predict(model,self.g.x_dict,self.g.edge_index_dict,pairs,baseline)
            richer=models.predict(model,self.g.x_dict,self.g.edge_index_dict,pairs,features)
            self.assertGreater(float(np.max(np.abs(richer-first))),1e-5)
            x=features.clone().requires_grad_(True)
            idx,visible=models.training_context(self.pairs,self.labels,self.g.edge_index_dict,
                self.weights,'LncDrug','lncRNA',self.g['drug'].num_nodes,.5,666,1)
            model.train()
            logits=model(self.g.x_dict,visible,pairs[:,idx],x[idx])
            torch.nn.functional.binary_cross_entropy_with_logits(logits,
                torch.from_numpy(self.labels[idx])).backward()
            self.assertTrue(bool(torch.isfinite(x.grad).all()))
            self.assertGreater(float(x.grad[:,36:].abs().sum()),0)
            for param in model.parameters():
                if param.grad is not None:
                    self.assertTrue(bool(torch.isfinite(param.grad).all()))

    def test_checkpoint_replay_query_order_and_rng_restoration(self):
        features=torch.tensor(np.random.default_rng(9).normal(size=(len(self.labels),62)),
                              dtype=torch.float32)
        pairs=torch.from_numpy(self.pairs)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.pt'
            for architecture in compare.ARCHITECTURES:
                model=self.make(architecture)
                expected=models.predict(model,self.g.x_dict,self.g.edge_index_dict,pairs,features)
                torch.save(model.state_dict(),path)
                before=torch.get_rng_state().clone()
                with torch.random.fork_rng(devices=[]):
                    replay=self.make(architecture)
                    replay.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
                    predictions=models.predict(replay,self.g.x_dict,self.g.edge_index_dict,pairs,features)
                    np.testing.assert_allclose(predictions,expected,atol=1e-6,rtol=1e-6)
                    reverse=models.predict(replay,self.g.x_dict,self.g.edge_index_dict,
                        pairs.flip(1),features.flip(0))[::-1]
                    split=np.concatenate([models.predict(replay,self.g.x_dict,self.g.edge_index_dict,
                        pairs[:,part],features[part]) for part in np.array_split(np.arange(len(features)),3)])
                    np.testing.assert_allclose(reverse,expected,atol=1e-6,rtol=1e-6)
                    np.testing.assert_allclose(split,expected,atol=1e-6,rtol=1e-6)
                    contribution=models.graph_contribution(replay,self.g.x_dict,
                        self.g.edge_index_dict,pairs,features)
                    if architecture=='sahgt':
                        self.assertTrue(np.isfinite(contribution['graph_to_pair_logit_std_ratio']))
                        self.assertAlmostEqual(contribution['graph_gate'],1/(1+np.exp(2)),places=6)
                torch.testing.assert_close(torch.get_rng_state(),before)

    def test_confident_predictions_retain_ranking(self):
        head=torch.nn.Linear(62,1,bias=False)
        with torch.no_grad():
            head.weight.zero_();head.weight[0,0]=1
        model=models.MatchedMLP(head)
        features=torch.zeros((3,62));features[:,0]=torch.tensor([30.,31.,32.])
        pairs=torch.from_numpy(self.pairs[:,:3])
        score=models.predict(model,self.g.x_dict,self.g.edge_index_dict,pairs,features)
        self.assertTrue(bool((np.diff(score)>0).all()))
        metrics=compare.base.metrics(np.array([0,1,1]),score,pairs.numpy())
        self.assertEqual(metrics['auc'],1.)
        self.assertEqual(metrics['ap'],1.)


class ArtifactControls(unittest.TestCase):
    def test_checkpoint_identity_and_artifact_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);(folder/'scores.npz').write_bytes(b'fixture')
            identity=dict(task='LncDrug',architecture='mlp',view='graph36',seed=0,
                          budget='historical',epoch=24)
            result=dict(identity,protocol_sha256='test',replay_passed=True,
                        artifacts={'scores.npz':compare.base.digest(folder/'scores.npz')})
            compare.base.write_json(folder/'result.json',result)
            self.assertEqual(compare.checked_checkpoint(folder,{'sha256':'test'},identity),result)
            with self.assertRaisesRegex(ValueError,'identity'):
                compare.checked_checkpoint(folder,{'sha256':'test'},dict(identity,epoch=25))
            (folder/'scores.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                compare.checked_checkpoint(folder,{'sha256':'test'},identity)


if __name__=='__main__':
    unittest.main()
