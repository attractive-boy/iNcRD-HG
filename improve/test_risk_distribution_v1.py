"""Independent weighted-MMD, higher-moment, data-boundary and budget checks."""
import inspect
import unittest
from unittest.mock import patch

import numpy as np
import torch

from improve import anchor_risk_distribution_v1 as experiment
from improve import risk_distribution_graph_v1 as core
from improve.test_nested_anchor_v2 import fixture


def training_fixture():
    a, _ = fixture()
    a['anchor_oof'] = np.linspace(-1., 1., len(a['y_train']))
    a['x_2'] = np.array([[1., .1, .2, .3]]*3+[[-1., -.1, .2, .3]]*3, dtype=np.float32)
    c = core.parameters()
    c.update(width=8, heads=2, domain_count=2, minimum_class_domain_samples=1,
             warmup_steps=2)
    p = dict(smoke=True, family='risk_distribution', implementation_module='risk_distribution_graph_v1',
             source_sha256={'risk_distribution_graph_v1.py':experiment.common.base.digest(core.__file__)},
             core_parameters=c, kinds=list(core.KINDS), epochs=7, check_interval=5,
             weight_decay=.001, strong_weight=.1)
    return a, p


class RiskDistributionTests(unittest.TestCase):
    def test_weighted_mmd_matches_independent_three_expectations_and_gradient(self):
        values = torch.tensor([.1, .4, 1.2, 2.3],dtype=torch.float64,requires_grad=True)
        p = torch.tensor([.5,.5,0,0],dtype=torch.float64)
        q = torch.tensor([.125,.125,.25,.5],dtype=torch.float64)
        gammas = [.1,1.,10.]
        def kernel(x,y):
            return sum(torch.exp(-g*(x-y).square()) for g in gammas)
        expected = sum((p[i]*p[j]+q[i]*q[j]-2*p[i]*q[j])*kernel(values[i],values[j])
                       for i in range(4) for j in range(4))
        actual = core.rbf_mmd(values,p-q,gammas)
        torch.testing.assert_close(actual,expected,atol=1e-13,rtol=1e-13)
        ga = torch.autograd.grad(actual,values,retain_graph=True)[0]
        ge = torch.autograd.grad(expected,values)[0]
        torch.testing.assert_close(ga,ge,atol=1e-12,rtol=1e-12)
        self.assertTrue(torch.autograd.gradcheck(lambda x:core.rbf_mmd(x,p-q,gammas),(values,)))
        ix = torch.tensor([2,0,3,1])
        torch.testing.assert_close(core.rbf_mmd(values[ix],(p-q)[ix],gammas),actual)

    def test_full_distribution_detects_equal_mean_equal_variance_difference(self):
        s = np.sqrt(2.)
        values = torch.tensor([1.,3.,1.,3.,2.,2.,2.-s,2.+s]*2,dtype=torch.float64,requires_grad=True)
        labels = torch.tensor([0]*8+[1]*8)
        domains = torch.tensor(([0]*4+[1]*4)*2)
        full,moment,info = core.matching_terms(values,labels,domains,core.parameters())
        self.assertGreater(float(full.detach()),.05)
        self.assertLess(abs(float(moment.detach())),1e-26)
        self.assertEqual(len(info),2)
        self.assertTrue(torch.isfinite(torch.autograd.grad(full,values)[0]).all())

    def test_domains_use_training_entities_and_attributes_without_outcomes(self):
        a,p = training_fixture()
        keep = a['pairs_train'][1] != 5
        a['pairs_train'] = a['pairs_train'][:,keep]
        for key in ('y_train','folds','anchor_oof'):
            a[key] = a[key][keep]
        domains = core.drug_domains(a,p['core_parameters'])
        self.assertEqual(domains[5],-1)
        b = {key:value.copy() for key,value in a.items()}
        b['y_train'] = 1-b['y_train']
        b['anchor_oof'] *= -100
        b['x_2'][5] = 1e8
        b['pairs_train'] = b['pairs_train'][:,::-1]
        np.testing.assert_array_equal(domains,core.drug_domains(b,p['core_parameters']))
        with self.assertRaisesRegex(ValueError,'training arrays only'):
            core.setup(dict(a,y_val=np.zeros(8)),p['core_parameters'],'full','cpu')

    def test_training_warmup_gradient_budget_and_rng_are_matched(self):
        a,p = training_fixture()
        initial_states, rng_states = [],[]
        for kind in core.KINDS:
            with patch('torch.manual_seed',side_effect=AssertionError('all-device seed forbidden')):
                model,value,trace,detail = experiment.fit(a,p,kind,.003,917,'cpu',fixed_epochs=7)
                warm,_,_,_ = experiment.fit(a,p,kind,.003,917,'cpu',fixed_epochs=2)
            self.assertIsNone(value)
            self.assertEqual((detail['training_steps'],detail['gradient_evaluations'],detail['optimizer_steps']),(7,7,7))
            self.assertEqual(detail['matching_active_steps'],5)
            self.assertEqual([r['matching_weight'] for r in trace],[0,0,1,1,1,1,1])
            self.assertTrue(all('anchor_composed' not in r for r in trace))
            self.assertTrue(all(int(v['step'])==7 for v in model.training_final_state['optimizer']['state'].values()))
            self.assertTrue(any(r['effective_penalty'] > 0 for r in trace[2:]) if kind != 'erm'
                            else all(r['effective_penalty']==0 for r in trace))
            initial_states.append(warm.state_dict())
            rng_states.append(model.training_final_state['cpu_rng'])
        for other in initial_states[1:]:
            for key,value in initial_states[0].items():
                torch.testing.assert_close(value,other[key],atol=0,rtol=0)
        for state in rng_states[1:]:
            torch.testing.assert_close(state,rng_states[0],atol=0,rtol=0)
        with self.assertRaisesRegex(ValueError,'training arrays only'):
            experiment.fit(dict(a,y_val=np.zeros(7)),p,'full',.003,917,'cpu',fixed_epochs=7)
        self.assertNotIn('torch.manual_seed(',inspect.getsource(experiment))

    def test_lnc_identity_has_no_domains_training_or_outcome_dependency(self):
        a,p = training_fixture()
        keep = a['pairs_train'][0] < int(a['node_counts'][0])
        a['pairs_train'] = a['pairs_train'][:,keep]
        for key in ('y_train','folds','anchor_oof'):
            a[key] = a[key][keep]
        a['pair_offset'] = np.array([0,14])
        for kind in core.KINDS:
            model,value,trace,detail = experiment.fit(a,p,kind,.003,917,'cpu',fixed_epochs=0)
            self.assertIsNone(value)
            self.assertFalse(model.state_dict())
            self.assertFalse(detail['trained_neural_model'])
            self.assertFalse(trace)
            self.assertEqual(detail['gradient_evaluations'],0)
            self.assertIsNone(model.training_final_state)
            _,value,_,_ = experiment.fit(dict(a,pairs_val=a['pairs_train'][:,:5]),p,kind,.003,917,'cpu')
            np.testing.assert_array_equal(value,np.zeros(5))


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
