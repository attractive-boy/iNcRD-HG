"""Rooted weighted mean: direct path enumeration, gradients and empty support."""
import json,itertools
import numpy as np
import torch
from common import HERE,sha256
from model import rooted_values,sym_normalize

def explicit(s,a,h,endpoints):
    num=0.;den=0.
    for di,ri in itertools.product(range(17,20),range(1,4)):
        for u,v,w,t in itertools.product([0,1],[2,3],[0,1],[2,3]):
            mass=a[0,di]*a[di,ri]*a[ri,16]*h[u,v]*h[v,w]*h[w,t]
            weight=s[u,0]*s[t,16]
            if not endpoints:weight*=s[v,di]*s[w,ri]
            num+=mass*weight;den+=mass
    return np.array([num/den,num,den])

def main():
    torch.manual_seed(42);dtype=torch.double
    s=torch.rand((1,1,4,32),dtype=dtype);s[:,:,:2,16:]=0;s[:,:,2:,:16]=0;s.requires_grad_()
    a=torch.zeros((1,32,32),dtype=dtype)
    for r,d in itertools.product(range(4),range(16,20)):
        if (r,d)!=(0,16):a[0,r,d]=a[0,d,r]=1
    a=sym_normalize(a)
    h=torch.zeros((1,4,4),dtype=dtype);h[:,:2,2:]=torch.rand(1,2,2,dtype=dtype)+.2;h[:,2:,:2]=h[:,:2,2:].transpose(-1,-2);h=sym_normalize(h).requires_grad_()
    rows=[]
    for endpoints in [False,True]:
        out=rooted_values(s,a,h,endpoints)
        ref=explicit(s.detach().numpy()[0,0],a.numpy()[0],h.detach().numpy()[0],endpoints)
        np.testing.assert_allclose(np.array([float(x) for x in out]),ref,atol=1e-12,rtol=1e-12)
        grad=torch.autograd.grad(out[0].sum(),s,retain_graph=True)[0]
        if endpoints:assert grad[:,:,:,1:16].abs().max()==0
        else:assert grad[:,:,:,1:4].abs().max()>1e-5
        # Input attribute finite difference at an actual intermediate vertex.
        index=(0,0,0,1);eps=1e-6;up=s.detach().clone();down=up.clone();up[index]+=eps;down[index]-=eps
        fd=(rooted_values(up,a,h,endpoints)[0]-rooted_values(down,a,h,endpoints)[0])/(2*eps)
        torch.testing.assert_close(fd.flatten()[0],grad[index],atol=1e-9,rtol=1e-6)
        # General H entries affect both numerator and denominator; check quotient gradient.
        assert torch.autograd.gradcheck(lambda w:rooted_values(s,a,w,endpoints)[0],(h,),eps=1e-6,atol=1e-5,rtol=1e-4)
        empty=rooted_values(s,torch.zeros_like(a),h,endpoints)
        for x in empty:assert torch.isfinite(x).all() and x.abs().max()==0
        assert 0<float(out[0])<1
        rows.append(dict(endpoints=endpoints,enumeration_error=float(abs(np.array([float(x) for x in out])-ref).max()),
            intermediate_gradient=float(grad[index]),finite_difference=True,hidden_graph_gradcheck=True,empty_support_exact_zero=True))
    (HERE/'math_checks.json').write_text(json.dumps(dict(status='passed',records=rows,source_hashes={str(p):sha256(p) for p in [HERE/'model.py',HERE/'common.py',HERE/'check_math.py']}),indent=2))
    print('Rooted math checks passed')

if __name__=='__main__':main()
