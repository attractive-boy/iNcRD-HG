"""The new control changes only feature-variance gradients; reuse stays exact."""
import inspect
import unittest

import numpy as np
import torch

from improve import variational_lastlayer_graph_v1 as old_core
from improve import variational_feature_learning_v1 as core
from improve import anchor_variational_lastlayer_experiment_v1 as parent
from improve import anchor_variational_feature_experiment_v1 as experiment
from improve import adopt_variational_feature_v1 as adoption
from improve.test_anchor_variational_lastlayer_experiment_v1 import protocol as old_protocol
from improve.test_nested_anchor_v2 import fixture


def protocol():
    p = old_protocol()
    p.update(family='variational_feature_learning',implementation_module='variational_feature_learning_v1',
             kinds=list(core.KINDS),source_sha256={'variational_feature_learning_v1.py':parent.common.base.digest(core.__file__)})
    return p


class VariationalFeatureTests(unittest.TestCase):
    def test_control_preserves_forward_and_head_gradients_only(self):
        ordinary = old_core.GaussianClassifier(3,'diagonal').double()
        control = core.DetachedFeatureVariance(3,'diagonal').double()
        with torch.no_grad():
            ordinary.mean.copy_(torch.arange(8,dtype=torch.float64).reshape(2,4)/13)
        control.load_state_dict(ordinary.state_dict())
        x = torch.tensor([[.3,-.7,1.1],[.5,.2,-.4]],dtype=torch.float64,requires_grad=True)
        z = x.detach().clone().requires_grad_()
        mean,var = ordinary(x)
        other_mean,other_var = control(z)
        torch.testing.assert_close(mean,other_mean,atol=0,rtol=0)
        torch.testing.assert_close(var,other_var,atol=0,rtol=0)
        (mean.sum()+var.sum()).backward()
        (other_mean.sum()+other_var.sum()).backward()
        self.assertGreater(float((x.grad-z.grad).norm()),0)
        torch.testing.assert_close(z.grad,ordinary.mean[:,:3].sum(0).expand_as(z),atol=0,rtol=0)
        for key in ('mean','log_scale'):
            torch.testing.assert_close(getattr(ordinary,key).grad,getattr(control,key).grad,atol=0,rtol=0)

    def test_reused_full_and_map_training_and_rng_are_exact(self):
        a,_ = fixture()
        a['anchor_oof'] = np.linspace(-2,2,len(a['y_train']))
        for kind,old_kind in (('full','diagonal'),('map','map')):
            new,_,trace,_ = experiment.fit(a,protocol(),kind,.001,17,'cpu',fixed_epochs=5)
            state = torch.random.get_rng_state().clone()
            old,_,prior_trace,_ = parent.fit(a,old_protocol(),old_kind,.001,17,'cpu',fixed_epochs=5)
            self.assertEqual(trace,prior_trace)
            self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
            for key,value in new.state_dict().items():
                torch.testing.assert_close(value,old.state_dict()[key],atol=0,rtol=0)

    def test_executor_bodies_and_adoption_recipe_guard(self):
        for name in ('fit','predict','predict_components','run'):
            self.assertEqual(inspect.getsource(getattr(parent,name)),inspect.getsource(getattr(experiment,name)))
        old = parent.common.protocol(parent.ROOT/'results/anchor_priority_variational_lastlayer_v1_20260916')
        adoption.verify_recipe(dict(old),old)
        with self.assertRaisesRegex(ValueError,'recipe'):
            adoption.verify_recipe(dict(old,epochs=399),old)

    def test_new_control_training_boundary_and_identity(self):
        a,_ = fixture()
        a['anchor_oof'] = np.zeros_like(a['y_train'],dtype=np.float64)
        p = protocol()
        with self.assertRaisesRegex(ValueError,'training arrays only'):
            experiment.fit(dict(a,y_val=np.ones(3)),p,'no_feature_variance',.001,17,'cpu',fixed_epochs=1)
        model,_,trace,detail = experiment.fit(a,p,'no_feature_variance',.001,17,'cpu',fixed_epochs=5)
        self.assertIsInstance(model.last,core.DetachedFeatureVariance)
        self.assertEqual(detail['training_steps'],5)
        self.assertTrue(all(np.isfinite(row['loss']) for row in trace))
        a['pair_offset'] = np.array([0,14])
        model,_,trace,detail = experiment.fit(a,p,'no_feature_variance',.001,17,'cpu',fixed_epochs=0)
        self.assertEqual(trace,[])
        self.assertEqual(list(model.parameters()),[])
        self.assertFalse(detail['trained_neural_model'])


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
