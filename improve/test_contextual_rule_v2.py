"""Larger synthetic graph for the training integration check.

The v1 8-by-6 parity fixture had no usable rule-ranking pair in a support
fold. Keep that failed record and retain the production guard unchanged.
"""
from unittest.mock import patch
import unittest

import numpy as np
import torch

from improve import test_contextual_rule_v1 as prior_tests
from improve import contextual_rule_graph_v1 as core
from improve import anchor_contextual_rule_v1 as experiment


def larger_fixture():
    counts = np.array([12,16,12,4])
    offsets = np.r_[0,np.cumsum(counts)[:-1]]
    pairs = np.indices((16,12)).reshape(2,-1)
    y = ((pairs[1]==pairs[0]%12)|(pairs[1]==(pairs[0]+1)%12)).astype(np.float32)
    edges,relations,codes = [],[],[]
    for relation,(left,right) in enumerate(core.parameters()['physical_schema']):
        if relation==3:
            selected = pairs[:,y==1]
        elif left==right:
            selected = np.array([np.arange(counts[left]-1),np.arange(1,counts[left])])
        else:
            n = min(counts[left],counts[right])
            selected = np.array([np.arange(n),np.arange(n)])
        edges.append(selected+offsets[[left,right],None])
        relations.append(np.full(selected.shape[1],relation,dtype=np.int64))
        codes.append(selected[0]*12+selected[1] if relation==3 else np.full(selected.shape[1],-1,dtype=np.int64))
    a = dict(pairs_train=pairs,y_train=y,folds=np.arange(len(y))%5,node_counts=counts,
             pair_offset=offsets[[1,2]],nd=np.array(12),edges=np.concatenate(edges,1),
             relation=np.concatenate(relations),target_codes=np.concatenate(codes),
             anchor_oof=np.linspace(-1.,1.,len(y)))
    rng = np.random.default_rng(9147)
    for i,n in enumerate(counts):
        a['x_'+str(i)] = rng.normal(size=(n,4)).astype(np.float32)
    return a


class ContextualRuleTests(prior_tests.ContextualRuleTests):
    def test_rule_triplets_and_actual_fit_budget(self):
        a = larger_fixture()
        c = prior_tests.settings()
        p = dict(smoke=True,family='contextual_rule',implementation_module='contextual_rule_graph_v1',
            source_sha256={'contextual_rule_graph_v1.py':experiment.common.base.digest(core.__file__)},
            core_parameters=c,kinds=list(core.KINDS),epochs=5,check_interval=20,weight_decay=.001,
            strong_weight=.1,private_rng_offset=29015)
        random_states = []
        for kind in core.KINDS:
            with patch('torch.manual_seed',side_effect=AssertionError('all-device seed forbidden')):
                model,value,trace,detail = experiment.fit(a,p,kind,.001,819,'cpu',fixed_epochs=5)
            self.assertIsNone(value)
            self.assertEqual((detail['training_steps'],detail['optimizer_steps'],detail['gradient_evaluations']),(5,5,5))
            self.assertTrue(all(row['rule_supervision_verified'] and row['triplet_count']==16 for row in trace))
            self.assertTrue(all('anchor_composed' not in row for row in trace))
            self.assertTrue(all(int(item['step'])==5 for item in model.training_final_state['optimizer']['state'].values()))
            random_states.append(model.training_final_state['private_rng'])
        torch.testing.assert_close(*random_states,atol=0,rtol=0)
        with self.assertRaisesRegex(ValueError,'training arrays only'):
            experiment.fit(dict(a,y_val=np.zeros(3)),p,'full',.001,819,'cpu',fixed_epochs=5)
        keep = a['pairs_train'][0]<int(a['node_counts'][0])
        a['pairs_train'] = a['pairs_train'][:,keep]
        for key in ('y_train','folds','anchor_oof'):
            a[key] = a[key][keep]
        a['pair_offset'] = np.array([0,28])
        model,value,trace,detail = experiment.fit(a,p,'full',.001,819,'cpu',fixed_epochs=0)
        self.assertIsNone(value)
        self.assertFalse(trace)
        self.assertFalse(model.state_dict())
        self.assertFalse(detail['trained_neural_model'])


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
