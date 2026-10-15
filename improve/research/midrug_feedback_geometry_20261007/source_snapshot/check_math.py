"""Independent NumPy recurrence, finite differences and physical graph deletions."""
import copy,json
import numpy as np
import torch
from torch_geometric.data import HeteroData
from common import HERE,sha256
from modules import GeometryFeedback


def linear(x,l):
    y=x@l.weight.detach().numpy().T
    if l.bias is not None:y+=l.bias.detach().numpy()
    return y


def reference(module,x,z,nx,nz,valid,types):
    b,m=module.blocks,module.block_size;ii=np.tril_indices(m,-1);o=len(ii[0])
    def frame(v):
        a=np.zeros((b,m,m));a[:,ii[0],ii[1]]=v[:,:o]
        a[:,np.arange(m),np.arange(m)]=np.exp(v[:,o:]);return a
    rows=[]
    for t in range(len(x)):
        L=frame(z[t]);h=np.einsum('bij,bj->bi',L,x[t].reshape(b,m));energies=[];messages=[]
        for j in range(nx.shape[1]):
            NL=frame(nz[t,j]);nh=np.einsum('bij,bj->bi',NL,nx[t,j].reshape(b,m))
            energies.append(-np.square(nh-h).sum())
            env=np.einsum('bij,bj->bi',NL,linear(nx[t,j],module.phi).reshape(b,m))
            messages.append(np.stack([np.linalg.solve(L[k],env[k]) for k in range(b)]))
        e=np.array(energies);e[~valid[t]]=-np.inf;w=np.exp(e-e.max());w/=w.sum();msgs=np.array(messages)
        agg=np.sum(w[:,None,None]*msgs,axis=0).reshape(-1)
        rate=1/(1+np.exp(-linear(np.r_[x[t],agg],module.gate)))
        newx=(1-rate)*x[t]+rate*agg
        residual=msgs-linear(x[t],module.phi).reshape(b,m)
        C=np.zeros((b,m,m))
        if module.mode=='mean':
            mean=np.sum(w[:,None,None]*residual,axis=0)
            for k in range(b):C[k]=np.outer(mean[k],mean[k])
        else:
            for j in range(len(w)):
                for k in range(b):C[k]+=w[j]*np.outer(residual[j,k],residual[j,k])
            if module.mode=='diagonal':C=np.array([np.diag(np.diag(c)) for c in C])
        C+=1e-3*np.eye(m);R=np.linalg.cholesky(C+1e-5*np.eye(m))
        target=np.concatenate([R[:,ii[0],ii[1]],np.log(np.diagonal(R,axis1=-2,axis2=-1))],axis=1)
        inp=np.concatenate([x[t].reshape(b,m),agg.reshape(b,m),np.broadcast_to(types[t],(b,2))],axis=1)
        ctrl=linear(inp,module.controller[0]);ctrl=ctrl/(1+np.exp(-ctrl));ctrl=linear(ctrl,module.controller[2])
        U=ctrl[:,:m*m].reshape(b,m,m);V=ctrl[:,m*m:2*m*m].reshape(b,m,m)
        mat=U@V.transpose(0,2,1)
        correction=np.concatenate([mat[:,ii[0],ii[1]],ctrl[:,2*m*m:]],axis=1)
        newz=.5*z[t]+.5*target+.1*correction
        newz[:,:o]=np.clip(newz[:,:o],-2,2);newz[:,o:]=np.clip(newz[:,o:],-5,5)
        rows.append((newx,newz,w,msgs,C))
    return tuple(np.stack([r[i] for r in rows]) for i in range(5))


def synthetic_graph():
    g=HeteroData();g['miRNA'].x=torch.randn(4,3,dtype=torch.double);g['drug'].x=torch.randn(3,5,dtype=torch.double)
    g['miRNA','MiMi','miRNA'].edge_index=torch.tensor([[0,0,1,2,3,0],[0,0,2,1,3,1]])
    g['drug','DrugDrug','drug'].edge_index=torch.tensor([[0,0,1,2],[0,1,1,2]])
    e=torch.tensor([[0,1,2,2],[0,1,0,2]])
    g['miRNA','MiDrug','drug'].edge_index=e;g['drug','rev_MiDrug','miRNA'].edge_index=e.flip(0)
    return g


def main():
    torch.set_num_threads(2);torch.manual_seed(42);rng=np.random.default_rng(42);records=[]
    for mode in ['full','diagonal','mean']:
        a=GeometryFeedback(3,5,mode,blocks=2,block_size=2).double()
        x=torch.tensor(rng.normal(0,.2,(3,4)),requires_grad=True)
        z=torch.tensor(rng.normal(0,.1,(3,2,3)),requires_grad=True)
        nx=torch.tensor(rng.normal(0,.2,(3,5,4)),requires_grad=True)
        nz=torch.tensor(rng.normal(0,.1,(3,5,2,3)),requires_grad=True)
        valid=torch.tensor([[1,1,0,1,0],[1,0,0,0,0],[1,1,1,1,1]],dtype=torch.bool)
        types=torch.tensor([[1.,0.],[0.,1.],[1.,0.]],dtype=torch.double)
        out=a.step(x,z,nx,nz,valid,types,True)
        values=[v.detach().numpy() for v in [x,z,nx,nz,valid,types]]
        expected=reference(a,*values)
        actual=[out[0],out[1],out[2]['weights'],out[2]['messages'],out[2]['moment']]
        for u,v in zip(actual,expected):np.testing.assert_allclose(u.detach(),v,atol=2e-11,rtol=2e-11)
        assert (out[2]['weights'][~valid]==0).all()
        objective=out[0].square().sum()+out[1].square().sum()
        grads=torch.autograd.grad(objective,[x,z,nx,nz,a.phi.weight,a.controller[0].weight])
        fd=[]
        for k,idx in [(0,(0,1)),(1,(0,0,0)),(2,(0,2,0)),(3,(0,1,0,0))]:
            pos=[v.copy() for v in values];neg=[v.copy() for v in values];pos[k][idx]+=1e-6;neg[k][idx]-=1e-6
            po=reference(a,*pos);ne=reference(a,*neg)
            numerical=(sum((v*v).sum() for v in po[:2])-sum((v*v).sum() for v in ne[:2]))/2e-6
            analytic=float(grads[k][idx]);assert abs(numerical-analytic)<2e-6
            fd.append(abs(numerical-analytic))
        # The invalid neighbor's value and frame gradients must both vanish.
        assert (grads[2][~valid]==0).all() and (grads[3][~valid]==0).all()
        g=synthetic_graph();a.prepare_graph(g);a.eval();pars=list(a.parameters())
        errors=[];gradient_errors=[]
        for pair in [[0,0],[1,1],[2,2],[3,2],[0,2]]:
            q=torch.tensor(pair)[:,None];p=a(g['miRNA'].x,g['drug'].x,q)
            # Rebuild a distinct graph object with target records physically absent.
            gg=copy.deepcopy(g);e=gg['miRNA','MiDrug','drug'].edge_index
            e=e[:,~((e[0]==pair[0])&(e[1]==pair[1]))]
            gg['miRNA','MiDrug','drug'].edge_index=e;gg['drug','rev_MiDrug','miRNA'].edge_index=e.flip(0)
            ref=copy.deepcopy(a);ref.prepare_graph(gg)
            x0,z0=ref.initial(gg['miRNA'].x,gg['drug'].x);xx,zz=x0,z0
            for _ in range(2):xx,zz=ref.full_step(xx,zz)
            h=(ref.frame(zz)@xx.reshape(-1,2,2,1)).squeeze(-1)
            pp=h[[pair[0],pair[1]+4]].reshape(1,8)
            torch.testing.assert_close(p,pp,atol=2e-10,rtol=2e-10)
            gp=torch.autograd.grad(p.square().sum(),pars);gr=torch.autograd.grad(pp.square().sum(),list(ref.parameters()))
            for u,v in zip(gp,gr):torch.testing.assert_close(u,v,atol=2e-9,rtol=2e-8)
            errors.append(float((p-pp).abs().max()));gradient_errors.append(max(float((u-v).abs().max()) for u,v in zip(gp,gr)))
        records.append(dict(mode=mode,numpy_max_error=max(float(abs(u.detach().numpy()-v).max()) for u,v in zip(actual,expected)),
            finite_difference_max_error=max(fd),physical_deletion_errors=errors,parameter_gradient_errors=gradient_errors))
    # Equal first moments/diagonals, different full second moments.
    a=GeometryFeedback(2,2,'full',blocks=1,block_size=2).double();a.phi.weight.data.copy_(torch.eye(2))
    x=torch.zeros(2,2,dtype=torch.double);z=torch.zeros(2,1,3,dtype=torch.double)
    nx=torch.tensor([[[0,0],[.3,.3],[-.3,-.3]],[[0,0],[.3,-.3],[-.3,.3]]],dtype=torch.double)
    nz=torch.zeros(2,3,1,3,dtype=torch.double);valid=torch.ones(2,3,dtype=torch.bool);types=torch.tensor([[1.,0.],[1.,0.]],dtype=torch.double)
    moments={}
    for mode in ['full','diagonal','mean']:
        a.mode=mode;C=a.step(x,z,nx,nz,valid,types,True)[2]['moment'];moments[mode]=float((C[0]-C[1]).abs().max())
    assert moments['full']>.01 and moments['diagonal']==moments['mean']==0
    result=dict(status='passed',records=records,moment_separation=moments,source_hashes={str(p):sha256(p) for p in [HERE/'modules.py',HERE/'check_math.py']},no_model_metrics=True)
    (HERE/'math_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
