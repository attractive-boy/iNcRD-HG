"""Check the named-pathway operator and its no-outcome normalization."""
import unittest

import numpy as np
import torch

from improve.pathway_operator import PathwayOperator, normalize_channels


class PathwayOperatorTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(13)
        self.a=dict(ur=rng.normal(size=(4,2)).astype('float32'),ud=rng.normal(size=(3,2)).astype('float32'),
                    y_train=np.array([0.,1.,0.,1.],dtype='float32'),
                    program_rna=rng.normal(size=(4,4,5)).astype('float32'),
                    program_drug=np.abs(rng.normal(size=(3,5))).astype('float32'))

    def test_operator_equals_explicit_pair_feature_dot_product(self):
        m=PathwayOperator(self.a,'context')
        with torch.no_grad():m.w.copy_(torch.arange(20).reshape(4,5)/20)
        q=torch.tensor([[0,2,3],[1,0,2]])
        extra=torch.zeros(q.shape[1])
        for c in range(4):extra+=((m.r[c][q[0]]*m.d[q[1]])*m.w[c]).sum(1)
        torch.testing.assert_close(m(q)-m.kernel(q),extra,atol=2e-6,rtol=2e-6)

    def test_missing_drug_program_returns_kernel_exactly(self):
        self.a['program_drug'][1]=0
        m=PathwayOperator(self.a,'context')
        with torch.no_grad():m.w.fill_(1.)
        q=torch.tensor([[0,1,2,3],[1,1,1,1]])
        torch.testing.assert_close(m(q),m.kernel(q),rtol=0,atol=0)

    def test_pair_covariance_is_psd_and_normalization_has_unit_energy(self):
        r,d,s=normalize_channels(self.a['program_rna'],self.a['program_drug'])
        f=np.concatenate([(c[:,None,:]*d[None,:,:]).reshape(-1,d.shape[1]) for c in r],axis=1)
        k=f@f.T
        self.assertGreater(np.linalg.eigvalsh(k).min(),-1e-5)
        np.testing.assert_allclose(np.mean(np.sum(f*f,axis=1)),len(r),rtol=1e-6)


if __name__=='__main__':unittest.main()
