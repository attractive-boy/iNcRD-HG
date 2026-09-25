"""Verify latent sampling equations, gradient ownership and training boundaries."""
import unittest

import numpy as np
import torch

from improve import stochastic_augmentation_graph_v1 as core
from improve import anchor_stochastic_augmentation_v1 as experiment
from improve.test_nested_anchor_v2 import fixture


def protocol():
    c = core.parameters()
    c.update(width=8,heads=2,score_hidden=16,triplet_batch=4)
    return dict(smoke=True,family='stochastic_augmentation',implementation_module='stochastic_augmentation_graph_v1',
        source_sha256={'stochastic_augmentation_graph_v1.py':experiment.common.base.digest(core.__file__)},
        core_parameters=c,kinds=list(core.KINDS),epochs=2,check_interval=20,weight_decay=.001,
        strong_weight=.1,private_rng_offset=29015)


class StochasticAugmentationTests(unittest.TestCase):
    def test_reverse_variance_increments_and_repeated_positive_injection(self):
        p = core.parameters()
        x = torch.tensor([[1.,2.],[-2.,-.5]],dtype=torch.float64)
        noise = torch.zeros((11,*x.shape),dtype=x.dtype)
        zero = lambda x,s:torch.zeros_like(x)
        _,full = core.reverse_sample(zero,x,p,noise,'full',inject=True)
        expected = x.clone()
        expected[1] = .9**10*x[1]+(1-.9**10)*x[0]
        torch.testing.assert_close(full,expected,atol=1e-14,rtol=1e-14)
        _,control = core.reverse_sample(zero,x,p,noise,'no_injection',inject=True)
        torch.testing.assert_close(control,x,atol=0,rtol=0)
        drift = lambda x,s:2*torch.ones_like(x)
        _,full = core.reverse_sample(drift,x,p,noise,'full')
        _,control = core.reverse_sample(drift,x,p,noise,'no_reverse_drift')
        sigma = core.sigma_at(torch.tensor([.1,0.],dtype=x.dtype),p)
        torch.testing.assert_close(full,x+2*(sigma[0]**2-sigma[1]**2),atol=1e-14,rtol=1e-14)
        torch.testing.assert_close(control,x,atol=0,rtol=0)

    def test_score_loss_detaches_features_but_reverse_updates_features(self):
        net = core.ScoreNetwork(8,16).double()
        x = torch.arange(32,dtype=torch.float64).reshape(4,8).div(17).requires_grad_()
        p = core.parameters()
        time = torch.tensor([.1,.2,.4,.8],dtype=x.dtype)
        noise = torch.ones_like(x)
        loss = core.denoising_loss(net,x,p,time,noise)
        loss.backward()
        self.assertIsNone(x.grad)
        self.assertTrue(any(v.grad is not None and v.grad.norm()>0 for v in net.parameters()))
        net.zero_grad(set_to_none=True)
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        _,sample = core.reverse_sample(net.score,x,p,torch.zeros((11,*x.shape),dtype=x.dtype),'full')
        sample.square().mean().backward()
        self.assertGreater(float(x.grad.norm()),0)
        self.assertTrue(all(v.grad is None for v in net.parameters()))

    def test_triplets_use_same_rna_and_only_the_current_training_fold(self):
        a,_ = fixture()
        pools = core.training_triplets(a,'cpu')
        generator = torch.Generator().manual_seed(192)
        for fold,pool in enumerate(pools):
            pos,neg = core.sample_triplets(pool,31,generator)
            np.testing.assert_array_equal(a['pairs_train'][0,pos],a['pairs_train'][0,neg])
            self.assertTrue(np.all(a['y_train'][pos]==1) and np.all(a['y_train'][neg]==0))
            self.assertTrue(np.all(a['folds'][pos]==fold) and np.all(a['folds'][neg]==fold))

    def test_two_optimizer_budget_training_isolation_and_identity(self):
        a,_ = fixture()
        a['anchor_oof'] = np.linspace(-1,1,len(a['y_train']))
        p = protocol()
        with self.assertRaisesRegex(ValueError,'training arrays only'):
            experiment.fit(dict(a,y_val=np.ones(3)),p,'full',.001,17,'cpu',fixed_epochs=1)
        with self.assertRaisesRegex(ValueError,'cuda:1'):
            experiment.fit(a,p,'full',.001,17,'cuda:0',fixed_epochs=1)
        for kind in core.KINDS:
            model,value,trace,detail = experiment.fit(a,p,kind,.001,17,'cpu',fixed_epochs=2)
            self.assertIsNone(value)
            self.assertEqual(detail['gradient_evaluations'],4)
            self.assertEqual(detail['score_optimizer_steps'],2)
            self.assertEqual(detail['encoder_optimizer_steps'],2)
            self.assertTrue(all('anchor_composed' not in row and np.isfinite(row['loss']) for row in trace))
            for name in ('encoder_optimizer','score_optimizer'):
                self.assertTrue(all(int(v['step'])==2 for v in model.training_final_state[name]['state'].values()))
            self.assertTrue(all(v.requires_grad for v in model.parameters()))
        a['pair_offset'] = np.array([0,14])
        model,_,trace,detail = experiment.fit(a,p,'full',.001,17,'cpu',fixed_epochs=0)
        self.assertFalse(detail['trained_neural_model'])
        self.assertEqual(trace,[])
        self.assertIsNone(model.training_final_state)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
