import unittest
import numpy as np
import torch
from improve.reification_kernel import TransferKernel,query_features


class FrozenGraphReadoutTests(unittest.TestCase):
    def test_no_graph_features_is_the_original_kernel(self):
        a=dict(ur=np.ones((3,2),dtype='float32'),ud=np.ones((2,2),dtype='float32'),
               y_train=np.array([1.,0.,0.],dtype='float32'))
        m=TransferKernel(a,np.empty((3,0),dtype='float32'))
        q=torch.tensor([[0,1,2],[0,1,0]])
        torch.testing.assert_close(m(q,torch.empty(3,0)),m.kernel(q),atol=0,rtol=0)

    def test_query_conditioning_has_no_candidate_batch_dependence(self):
        torch.manual_seed(4)
        states=torch.randn(3,12,8)
        known=torch.tensor([8,9,10]);rel=torch.tensor([11,11,11])
        full=query_features(states,known,rel,0,7)
        order=torch.tensor([2,0,1])
        moved=query_features(states[order],known[order],rel[order],0,7)
        torch.testing.assert_close(full[order],moved,atol=0,rtol=0)
        self.assertEqual(full.shape,(3,7,24))

    def test_normalization_is_fixed_after_fitting_inputs(self):
        a=dict(ur=np.eye(3,dtype='float32'),ud=np.eye(2,dtype='float32'),
               y_train=np.array([1.,0.,1.],dtype='float32'))
        train=np.array([[1.,3.],[2.,3.],[3.,3.]],dtype='float32')
        m=TransferKernel(a,train)
        original=m.mean.clone(),m.scale.clone()
        m(torch.tensor([[0],[1]]),torch.tensor([[1000.,-5000.]]))
        torch.testing.assert_close(m.mean,original[0],atol=0,rtol=0)
        torch.testing.assert_close(m.scale,original[1],atol=0,rtol=0)
        self.assertGreaterEqual(float(m.scale.min()),.001-1e-8)


if __name__=='__main__':unittest.main()
