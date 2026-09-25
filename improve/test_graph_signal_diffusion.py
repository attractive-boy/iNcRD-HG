import unittest
import numpy as np
import torch
from improve.graph_signal_diffusion import spectrum,SignalDiffusion


class GraphSignalDiffusionTests(unittest.TestCase):
    def test_spectrum_uses_only_passed_support_and_handles_isolated_nodes(self):
        a=np.array([[1,0,0],[0,1,0],[0,0,0]],dtype='float32')
        q,v=spectrum(a,np.eye(3),rank=3)
        self.assertTrue(np.isfinite(q).all() and np.isfinite(v).all())
        np.testing.assert_allclose((q*v)@q.T,np.diag([1,1,0]),atol=1e-6)

    def test_forward_and_reverse_equations(self):
        m=SignalDiffusion(3,np.eye(3,dtype='float32'),np.array([1,.5,0],dtype='float32'),width=4)
        x=torch.tensor([[1.,2.,3.]])
        torch.testing.assert_close(m.smooth(x),torch.tensor([[1.,1.,0.]]))
        torch.testing.assert_close(m.blur(x,0.),x)
        # For an exact denoiser, the deterministic inverse restores the original signal.
        m.reconstruct=lambda z,c,ac,t:x.expand_as(z)
        torch.testing.assert_close(m.predict(x,True),x)

    def test_query_row_batching_and_reload(self):
        torch.manual_seed(5)
        m=SignalDiffusion(3,np.eye(3,dtype='float32'),np.array([1,.5,0],dtype='float32'),width=4).eval()
        x=torch.randn(5,3)
        with torch.no_grad():
            full=m.predict(x);parts=torch.cat([m.predict(p) for p in torch.tensor_split(x,3)])
        torch.testing.assert_close(full,parts,atol=1e-6,rtol=1e-6)
        m2=SignalDiffusion(3,np.eye(3,dtype='float32'),np.array([1,.5,0],dtype='float32'),width=4).eval()
        m2.load_state_dict(m.state_dict())
        torch.testing.assert_close(m.predict(x),m2.predict(x))


if __name__=='__main__':unittest.main()
