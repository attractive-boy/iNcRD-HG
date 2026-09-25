"""Exact path enumeration, support isolation and contextual scorer checks."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from improve import contextual_rule_graph_v1 as core
from improve.test_nested_anchor_v2 import fixture


def settings():
    p = core.parameters()
    p.update(width=8,dropout=0.,minimum_rule_support=1,triplet_batch=16)
    return p


class ContextualRuleTests(unittest.TestCase):
    def test_simple_path_corrections_match_exhaustive_vertex_distinct_walks(self):
        a,_ = fixture()
        p = settings()
        mats,_,_ = core.relation_matrices(a,p,0)
        schema = core.schemas(p)
        offsets = np.r_[0,np.cumsum(a['node_counts'])[:-1]]
        for path in core.rule_vocabulary(p):
            expected = np.zeros((8,6),dtype=np.float32)
            for head in range(8):
                def walk(node,step,seen):
                    if step==len(path):
                        expected[head,node-int(offsets[2])] += 1
                        return
                    relation = path[step]
                    left,right = schema[relation]
                    local = node-int(offsets[left])
                    for other in mats[relation].getrow(local).indices+offsets[right]:
                        if int(other) not in seen:
                            walk(int(other),step+1,seen|{int(other)})
                global_head = int(offsets[1]+head)
                walk(global_head,0,{global_head})
            np.testing.assert_array_equal(core.simple_grounding(path,mats,schema),expected)

    def test_query_fold_outcomes_cannot_change_context_paths_or_statistics(self):
        a,_ = fixture()
        p = settings()
        for fold in range(5):
            graph = core.build_support(a,p,fold,'cpu')
            changed = {key:value.copy() for key,value in a.items()}
            mask = a['folds']==fold
            changed['y_train'][mask] = 1-changed['y_train'][mask]
            other = core.build_support(changed,p,fold,'cpu')
            self.assertEqual(graph['audit'],other['audit'])
            for key in graph:
                if key!='audit':
                    torch.testing.assert_close(graph[key],other[key],atol=0,rtol=0)
            mats,_,_ = core.relation_matrices(a,p,fold)
            queries = a['pairs_train'][:,mask]
            self.assertFalse(np.asarray(mats[3][queries[0],queries[1]]).any())

    def test_global_context_is_head_independent_and_null_is_exact(self):
        a,_ = fixture()
        p = settings()
        graph = core.build_support(a,p,0,'cpu')
        torch.random.default_generator.manual_seed(815)
        full,global_model = core.ContextualRules(p,'full'),core.ContextualRules(p,'global_rules')
        global_model.load_state_dict(full.state_dict())
        query = torch.as_tensor(a['pairs_train']+a['pair_offset'][:,None])
        for model in (full,global_model):
            model.eval()
            torch.testing.assert_close(model([],graph,query),torch.zeros(query.shape[1],dtype=torch.float64),atol=0,rtol=0)
        torch.nn.init.normal_(full.fc2.weight,std=.15)
        global_model.load_state_dict(full.state_dict())
        first,second = full.rule_scores(graph),global_model.rule_scores(graph)
        torch.testing.assert_close(second,second[:1].expand_as(second),atol=1e-7,rtol=1e-7)
        self.assertGreater(float((first-first[:1]).abs().max().detach()),1e-5)
        value = full([],graph,query)
        torch.testing.assert_close(value,full([],graph,query.flip(1)).flip(0),atol=0,rtol=0)
        loss = value.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(full.layers[0].weight.grad).all())

    def test_rule_triplets_and_actual_fit_budget(self):
        from improve import anchor_contextual_rule_v1 as experiment
        a,_ = fixture()
        a['anchor_oof'] = np.linspace(-1.,1.,len(a['y_train']))
        c = settings()
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
            random_states.append(model.training_final_state['private_rng'])
        torch.testing.assert_close(*random_states,atol=0,rtol=0)
        with self.assertRaisesRegex(ValueError,'training arrays only'):
            experiment.fit(dict(a,y_val=np.zeros(3)),p,'full',.001,819,'cpu',fixed_epochs=5)
        keep = a['pairs_train'][0]<6
        a['pairs_train'] = a['pairs_train'][:,keep]
        for key in ('y_train','folds','anchor_oof'):
            a[key] = a[key][keep]
        a['pair_offset'] = np.array([0,14])
        model,value,trace,detail = experiment.fit(a,p,'full',.001,819,'cpu',fixed_epochs=0)
        self.assertIsNone(value)
        self.assertFalse(trace)
        self.assertFalse(model.state_dict())
        self.assertFalse(detail['trained_neural_model'])


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
