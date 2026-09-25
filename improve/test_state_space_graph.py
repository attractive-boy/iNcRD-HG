import unittest

import numpy as np
import torch

from improve.state_space_graph import state_scan,scan_order,SelectiveMemory


def sequential(x,loga,b,c):
    state=x.new_zeros((x.shape[1],b.shape[-1],x.shape[-1]));values=[]
    for i in range(len(x)):
        state=loga[i].exp()[:,None,None]*state+b[i,:,:,None]*x[i,:,None,:]
        values.append((c[i,:,:,None]*state).sum(1))
    return torch.stack(values)


class StateSpaceTest(unittest.TestCase):
    def test_chunked_recurrence_values_and_all_gradients(self):
        torch.manual_seed(482)
        tensors=[torch.randn(19,2,3,dtype=torch.float64),
                 -torch.rand(19,2,dtype=torch.float64),
                 torch.randn(19,2,4,dtype=torch.float64),
                 torch.randn(19,2,4,dtype=torch.float64)]
        tensors=[x.requires_grad_() for x in tensors]
        expected=sequential(*tensors);weight=torch.randn_like(expected)
        gradient=torch.autograd.grad((expected*weight).sum(),tensors)
        for chunk in (1,4,8,32):
            actual=state_scan(*tensors,chunk=chunk)
            torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-11)
            grad=torch.autograd.grad((actual*weight).sum(),tensors)
            for a,b in zip(grad,gradient):torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)

    def test_state_crosses_chunk_boundary_and_reset_is_finite(self):
        x=torch.zeros(70,1,1);x[0]=1.
        loga=torch.full((70,1),-.01);b=c=torch.ones(70,1,1)
        actual=state_scan(x,loga,b,c,chunk=16)
        self.assertGreater(float(actual[64]),.5)
        loga[32]=-1000.
        reset=state_scan(x,loga,b,c,chunk=16)
        self.assertTrue(torch.isfinite(reset).all())
        self.assertEqual(float(reset[33]),0.)

    def test_degree_priority_and_shared_tie_noise(self):
        counts=np.log1p(np.array([[1,0],[0,0],[1,2],[0,1]],dtype=np.float32))
        order=scan_order(counts,20260915)
        np.testing.assert_array_equal(np.rint(np.expm1(counts)).sum(1)[order],[0,1,1,3])
        self.assertEqual(len(set(scan_order(counts,20260915,True))),4)

    def test_selective_parameters_receive_gradients(self):
        torch.manual_seed(51)
        module=SelectiveMemory(16,2,4,chunk=8).double()
        h=torch.randn(23,16,dtype=torch.float64,requires_grad=True)
        module(h).square().mean().backward()
        for name in ('input.weight','bc.weight','dt.weight','log_a','output.weight'):
            gradient=dict(module.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(gradient).all(),name)
            self.assertGreater(float(gradient.abs().sum()),0.,name)


if __name__=='__main__':unittest.main()
