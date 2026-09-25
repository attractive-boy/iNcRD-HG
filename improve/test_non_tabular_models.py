"""Checks for label exclusion, positive-semidefinite priors and pair operations."""
import unittest
import numpy as np
import torch

from improve.non_tabular_models import (feature_kernel, spectral_features,
    graph_operator, masked_contexts, KernelLogistic, GraphFactorization,
    AssociationDenoiser)


class NonTabularBoundaryTests(unittest.TestCase):
    def test_heldout_label_changes_cannot_change_its_context(self):
        pairs=np.array([[0,0,1,1,2,2],[0,1,0,1,0,1]])
        labels=np.array([1,0,1,0,1,0])
        folds=np.array([0,1,0,1,0,1])
        a=masked_contexts(pairs,labels,(3,2),folds)
        changed=labels.copy();changed[folds==0]=1-changed[folds==0]
        b=masked_contexts(pairs,changed,(3,2),folds)
        np.testing.assert_array_equal(a[0][1],b[0][1])
        np.testing.assert_array_equal(a[0][2],b[0][2])
        for f,pos,zero in a:
            q=pairs[:,folds==f]
            self.assertEqual(float(pos[q[0],q[1]].sum()+zero[q[0],q[1]].sum()),0.)

    def test_node_kernel_and_graph_prior_psd(self):
        x=np.random.default_rng(4).normal(size=(9,5));x[0]=0.
        k=feature_kernel(x)
        self.assertGreater(np.linalg.eigvalsh(k).min(),-1e-10)
        u=spectral_features(k,9)
        np.testing.assert_allclose(u@u.T,k,atol=2e-6)
        s=graph_operator(k,3)
        self.assertGreater(np.linalg.eigvalsh(np.eye(9)-s).min(),-1e-6)
        np.testing.assert_array_equal(np.diag(s),0.)

    def test_product_rkhs_scores_equal_explicit_tensor_features(self):
        rng=np.random.default_rng(8)
        r=rng.normal(size=(4,3)).astype('float32')
        d=rng.normal(size=(5,2)).astype('float32')
        m=KernelLogistic(r,d,.5)
        with torch.no_grad():m.w.copy_(torch.tensor(rng.normal(size=(3,2))))
        pairs=torch.tensor([[0,3,1,2],[4,0,2,1]])
        explicit=np.einsum('ni,nj->nij',r[pairs[0]],d[pairs[1]]).reshape(4,-1)
        expected=explicit@m.w.detach().numpy().reshape(-1)
        np.testing.assert_allclose(m(pairs).detach().numpy(),expected,atol=1e-6)
        np.testing.assert_allclose(explicit@explicit.T,
            (r[pairs[0]]@r[pairs[0]].T)*(d[pairs[1]]@d[pairs[1]].T),atol=5e-6)

    def test_factor_pair_readout_and_reordering(self):
        m=GraphFactorization(np.eye(3,dtype='float32'),np.eye(4,dtype='float32'),.3,2)
        p=torch.tensor([[2,0,1],[1,2,3]])
        torch.testing.assert_close(m(p),m.full()[p[0],p[1]])
        torch.testing.assert_close(m(p.flip(1)).flip(0),m(p))
        self.assertGreaterEqual(m.penalty(1.,1.,10.).item(),0.)

    def test_denoiser_replays_without_queries_and_both_axes_receive_gradients(self):
        torch.manual_seed(11)
        r=np.ones((3,2),dtype='float32');d=np.ones((4,2),dtype='float32')
        m=AssociationDenoiser(r,d,np.eye(3,dtype='float32'),np.eye(4,dtype='float32'),.3,8)
        a=torch.zeros(3,4);a[0,1]=1.;n=torch.zeros_like(a);n[1,0]=1.
        score=m.full(a,n,.5)
        score.sum().backward()
        self.assertGreater(m.row_encoder[0].weight.grad.abs().sum().item(),0.)
        self.assertGreater(m.col_encoder[0].weight.grad.abs().sum().item(),0.)
        m.eval()
        with torch.no_grad():
            first=m.full(a,n,.5);second=m.full(a,n,.5)
        torch.testing.assert_close(first,second,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
