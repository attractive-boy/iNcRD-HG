"""Direct product-graph sums, gradients, permutation and topology sensitivity."""
import itertools,json
import numpy as np
import torch
from common import HERE,sha256
from model import kernel_values,sym_normalize

def enumerate_walks(s,a,h,mode):
    shape=s.shape;v=s.reshape(-1);t=np.kron(h,a);out=[float(v@v)]
    if mode=='no_edges':t=np.eye(len(v))
    for length in range(1,4):
        total=0.
        for path in itertools.product(range(len(v)),repeat=length+1):
            value=v[path[0]]*v[path[-1]]
            for j in range(length):value*=t[path[j+1],path[j]]
            if mode!='endpoints':
                for j in range(1,length):value*=v[path[j]]
            total+=value
        out.append(total)
    return np.array(out)

def main():
    torch.manual_seed(42);dtype=torch.double
    s=torch.rand(1,1,2,3,dtype=dtype,requires_grad=True)
    a=torch.tensor([[[0.,1.,.4],[1.,0.,.7],[.4,.7,0.]]],dtype=dtype);a=sym_normalize(a)
    h=sym_normalize(torch.tensor([[[0.,.8],[.8,0.]]],dtype=dtype)).requires_grad_()
    rows=[];values={}
    for mode in ['pathwise','endpoints','fixed','no_edges']:
        out=kernel_values(s,a,h,mode)[0,0];ref=enumerate_walks(s.detach().numpy()[0,0],a.numpy()[0],h.detach().numpy()[0],mode)
        np.testing.assert_allclose(out.detach().numpy(),ref,atol=1e-12,rtol=1e-12)
        assert torch.autograd.gradcheck(lambda u,w:kernel_values(u,a,w,mode),(s,h),eps=1e-6,atol=1e-5,rtol=1e-4)
        values[mode]=out.detach();rows.append(dict(mode=mode,max_enumeration_error=float(abs(out.detach().numpy()-ref).max()),gradcheck=True))
    torch.testing.assert_close(values['pathwise'][:2],values['endpoints'][:2],atol=0,rtol=1e-14)
    assert (values['pathwise'][2:]-values['endpoints'][2:]).abs().max()>1e-4
    order=torch.tensor([2,0,1]);permuted=kernel_values(s[:,:,:,order],a[:,order][:,:,order],h,'pathwise')
    torch.testing.assert_close(permuted,kernel_values(s,a,h,'pathwise'),atol=1e-12,rtol=1e-12)
    # Two different 2-regular bipartite topologies with heterogeneous attributes.
    c=np.zeros((8,8));d=np.zeros_like(c)
    for i in range(4):
        for j in [i,(i+1)%4]:c[i,4+j]=c[4+j,i]=1
        for j in ([0,1] if i<2 else [2,3]):d[i,4+j]=d[4+j,i]=1
    assert np.array_equal(c.sum(0),d.sum(0));ss=torch.rand(1,1,2,8,dtype=dtype)
    tc=sym_normalize(torch.tensor(c,dtype=dtype)[None]);td=sym_normalize(torch.tensor(d,dtype=dtype)[None])
    delta=float((kernel_values(ss,tc,h,'pathwise')-kernel_values(ss,td,h,'pathwise')).abs().max());assert delta>1e-5
    out=dict(status='passed',records=rows,matching_p0_p1=True,interior_matching_changes_p2_p3=True,node_permutation_passed=True,
        same_degrees_topology_example_difference=delta,scope='Checks formulas and one heterogeneous-feature example, not universal graph separation.',
        source_hashes={str(p):sha256(p) for p in [HERE/'model.py',HERE/'common.py',HERE/'check_math.py']})
    (HERE/'math_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
