import unittest

import numpy as np
import torch

from improve.multiscale_transport import filtration_pair,gw_tensor,sinkhorn,transport


class TransportTest(unittest.TestCase):
    def test_rectangular_conservation(self):
        cost=torch.tensor([[0.,1.,2.],[3.,1.,0.]],dtype=torch.float64)
        p=torch.tensor([.3,.7],dtype=torch.float64)
        q=torch.tensor([.2,.35,.45],dtype=torch.float64)
        plan,error,_=sinkhorn(cost,p,q,.3,steps=500,tolerance=1e-12)
        torch.testing.assert_close(plan.sum(1),p,atol=1e-11,rtol=1e-11)
        torch.testing.assert_close(plan.sum(0),q,atol=1e-11,rtol=1e-11)
        self.assertLess(error,1e-11)

    def test_gw_distortion_against_four_index_definition(self):
        cs=torch.tensor([[0.,1.],[1.,0.]],dtype=torch.float64)
        ct=torch.tensor([[0.,.2,1.],[.2,0.,.7],[1.,.7,0.]],dtype=torch.float64)
        t=torch.tensor([[.1,.05,.15],[.1,.3,.3]],dtype=torch.float64)
        explicit=((cs[:,None,:,None]-ct[None,:,None,:]).square()*t[:,:,None,None]*t[None,None,:,:]).sum()
        calculated=(gw_tensor(cs,ct,t,t.sum(1),t.sum(0))*t).sum()
        torch.testing.assert_close(calculated,explicit)

    def test_filtration_changes_scale_without_self_edges(self):
        a=np.array([[1,1,0,0],[1,0,0,0],[0,1,1,0],[0,0,0,1]],dtype=np.float32)
        cs,ct,w=filtration_pair(a,a,3)
        self.assertGreater(len(w),1)
        self.assertAlmostEqual(float(w.sum()),1.)
        self.assertTrue(np.all(np.diagonal(cs,axis1=1,axis2=2)==0))
        self.assertFalse(np.array_equal(cs[0],cs[-1]))
        np.testing.assert_array_equal(cs,ct)

    def test_independent_permutations_and_query_independence(self):
        rng=np.random.default_rng(713)
        a=(rng.random((5,6))>.55).astype(np.float32)
        b=(rng.random((7,6))>.5).astype(np.float32)
        bridge=(rng.random((5,7))>.8).astype(np.float32)
        r,s=rng.permutation(5),rng.permutation(7)
        value,plan,_=transport(a,b,bridge,outer_steps=12)
        permuted,tp,_=transport(a[r],b[s],bridge[r][:,s],outer_steps=12)
        np.testing.assert_allclose(value[r],permuted,atol=2e-6,rtol=2e-6)
        np.testing.assert_allclose(plan[r][:,s],tp,atol=2e-6,rtol=2e-6)


if __name__=='__main__':
    unittest.main()
