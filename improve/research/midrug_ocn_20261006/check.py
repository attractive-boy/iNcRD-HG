"""Independent dense-power and PyG deletion checks before training."""
import json,time
import numpy as np
import torch
from common import HERE,load_inner,seed_all
from features import CONFIGS,adjacency,coefficients,transform
from model import make_adapter,corrected_pool
import check_modules

@torch.no_grad()
def explicit(A,pair):
    r,d=map(int,pair);nr,nd=A.shape;d+=nr
    B=torch.zeros(nr+nd,nr+nd,dtype=A.dtype,device=A.device)
    B[:nr,nr:]=A;B[nr:,:nr]=A.T;B[r,d]=0;B[d,r]=0
    powers={1:B,2:B@B,3:B@B@B}
    raw=[];mass=[]
    for k in (2,3):
        w=powers[k-1][r]*powers[k][d]+powers[k][r]*powers[k-1][d]+powers[k][r]*powers[k][d]
        w[r]=0;w[d]=0;raw.append(w);mass.append(powers[k].sum(1))
    return torch.stack(raw)[None],torch.stack(mass)[None]

def main():
    data=load_inner(0);A=adjacency(data,'cuda:0');rows=[]
    ix=np.r_[np.flatnonzero(data['train_y']==1)[[0,77,511]],np.flatnonzero(data['train_y']==0)[[0,77,511]]]
    for i in ix:
        pair=data['train_pairs'][i:i+1];raw,mass=coefficients(A,pair);er,em=explicit(A,pair[0])
        torch.testing.assert_close(raw,er,atol=0,rtol=0);torch.testing.assert_close(mass,em,atol=0,rtol=0)
        deleted=A.clone();deleted[tuple(pair[0])]=0;dr,dm=coefficients(deleted,pair)
        torch.testing.assert_close(raw,dr,atol=0,rtol=0);torch.testing.assert_close(mass,dm,atol=0,rtol=0)
        for name,cfg in CONFIGS.items():
            w,s=transform(raw,mass,cfg);ew,es=transform(er,em,cfg)
            torch.testing.assert_close(w,ew,atol=0,rtol=0);torch.testing.assert_close(s,es,atol=0,rtol=0)
            assert w[0,:,pair[0,0]].eq(0).all() and w[0,:,605+pair[0,1]].eq(0).all()
            if cfg['orthogonalize']:
                dot=torch.sum(w[:,0].double()*w[:,1].double(),1)
                assert dot.abs().max()<1e-8
        rows.append(dict(index=int(i),label=int(data['train_y'][i]),independent_dense_powers_exact=True,record_deletion_exact=True))
    seed_all(42);base=make_adapter(data,'control',0,'cuda:0');base.model.eval()
    seed_all(42);a=make_adapter(data,'orth_norm',0,'cuda:0');a.model.eval()
    assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in k.lower() for k in a.model.state_dict())
    with torch.no_grad():
        p0,p1=base.predict('val'),a.predict('val')
        a.model.ocn.head[-1].weight.normal_(std=.01)
    np.testing.assert_allclose(p0,p1,rtol=2e-5,atol=2e-5)
    checks=check_modules.check(a,count=6)
    # The new arbitrary intermediate-node pooling must equal explicit PyG states.
    pooling=[]
    with torch.no_grad():
        for i in ix[:3]:
            pair=torch.as_tensor(data['train_pairs'][i:i+1].T,device='cuda:0');r,d=pair
            zr,zd,_,_,corr=a.model.encoder.forward_dense(*a.dense,*pair,a.ps.qvals(*pair))
            w,_=a.model.cache.select(*pair);pr,pd=corrected_pool(w,r,d,zr,zd,corr,a.dense)
            g=a.graph.clone()
            for edge,first,second in [(('miRNA','MiDrug','drug'),r,d),(('drug','rev_MiDrug','miRNA'),d,r)]:
                e=g[edge].edge_index;g[edge].edge_index=e[:,~((e[0]==first)&(e[1]==second))]
            z=a.model.encoder(g.x_dict,g.edge_index_dict)
            torch.testing.assert_close(pr,w[:,:,:605]@z['miRNA'],atol=3e-4,rtol=3e-5)
            torch.testing.assert_close(pd,w[:,:,605:]@z['drug'],atol=3e-4,rtol=3e-5)
            pooling.append(int(i))
    a.model.train();opt=torch.optim.RMSprop(a.model.parameters(),lr=1e-4);times=[]
    for step in range(3):
        torch.cuda.synchronize();start=time.time();opt.zero_grad()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(a.logits('train'),a.y);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.model.parameters())
        assert a.model.ocn.rproj.weight.grad.abs().max()>0
        opt.step();torch.cuda.synchronize();times.append(time.time()-start)
    result=dict(status='passed',independent_coefficients=rows,initial_control_max_error=float(abs(p0-p1).max()),
        full_logit_deletion=checks,explicit_pyg_pooling=pooling,finite_nonzero_gradients=True,step_seconds=times,
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'initial_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()

