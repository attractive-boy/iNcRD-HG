"""Independent path enumeration and complete query-deleted prediction checks."""
import json,time,gc
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from common import HERE,DEGREE,read_module,Adapter,load_inner,seed_all
from model import CONFIGS,rectangle_sum,completion_fields,query_fields,make_adapter,configure,mask_generator,auxiliary_loss
from incrd_hg_m123.config import ModelConfig

QUERY_CHECK_SOURCE=DEGREE/'check.py'
checker=read_module('graph_completion_deletion_helper',QUERY_CHECK_SOURCE);checker.make_adapter=make_adapter
deletion_check=checker.deletion_check

def enumeration(A,K,P,r,d,mode):
    if mode=='direct':return np.zeros(3)
    U=(~K)*(P if mode=='soft' else .5)
    i,j=np.indices(A.shape);valid=(i!=r)&(j!=d);i,j=i[valid],j[valid]
    return np.log1p(np.array([(U[r,j]*A[i,j]*A[i,d]).sum(),
        (A[r,j]*U[i,j]*A[i,d]).sum(),(A[r,j]*A[i,j]*U[i,d]).sum()]))

def operator_checks():
    seed_all(122);factors=[torch.rand(4,3,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    X,Y,Z=factors;got=rectangle_sum(X,Y,Z);errors=[]
    for r in range(4):
        for d in range(3):
            truth=sum(X[r,j]*Y[i,j]*Z[i,d] for i in range(4) for j in range(3) if i!=r and j!=d)
            torch.testing.assert_close(got[r,d],truth,atol=1e-12,rtol=1e-12)
            ga=torch.autograd.grad(got[r,d],factors,retain_graph=True)
            gb=torch.autograd.grad(truth,factors,retain_graph=True)
            for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12);assert abs(float(a[r,d]))<1e-12
            errors.append(float(abs(got[r,d]-truth)))
    A=torch.tensor([[1.,1.,0.],[0.,1.,1.],[0.,0.,0.],[1.,0.,0.]],dtype=torch.float64)
    K=A.bool();K[2,2]=True;P=torch.rand_like(A,requires_grad=True);records=[]
    for mode in ['direct','uniform','soft']:
        f=completion_fields(A,K,P,mode);assert not f.requires_grad
        for r in range(4):
            for d in range(3):
                expected=enumeration(A.numpy(),K.numpy(),P.detach().numpy(),r,d,mode)
                np.testing.assert_allclose(f[r,d].numpy(),expected,atol=1e-12,rtol=1e-12)
                B=A.clone();J=K.clone();B[r,d]=0;J[r,d]=False
                other=completion_fields(B,J,P,mode)[r,d]
                torch.testing.assert_close(f[r,d],other,atol=1e-12,rtol=1e-12)
                records.append(dict(mode=mode,pair=[r,d],known=bool(K[r,d]),positive=bool(A[r,d]),error=float(abs(f[r,d]-other).max())))
    for mode in ['uniform','soft']:
        zero=completion_fields(A*0,K,P,mode);torch.testing.assert_close(zero,zero*0,atol=0,rtol=0)
    # Arbitrary row/column changes emulate first-layer endpoint changes.
    repairs=[]
    for r in range(4):
        for d in range(3):
            row=torch.rand(1,3,dtype=A.dtype);col=torch.rand(1,4,dtype=A.dtype)
            ri,di=torch.tensor([r]),torch.tensor([d])
            got=query_fields(A,K,P,row,col,ri,di,'soft')[0]
            B=A.clone();J=K.clone();B[r,d]=0;J[r,d]=False
            Q=P.detach().clone();Q[r]=row[0];Q[:,d]=col[0]
            truth=enumeration(B.numpy(),J.numpy(),Q.numpy(),r,d,'soft')
            np.testing.assert_allclose(got.numpy(),truth,atol=1e-12,rtol=1e-12)
            repairs.append(dict(pair=[r,d],max_error=float(abs(got.numpy()-truth).max())))
    return dict(max_weighted_sum_error=max(errors),cases=records,row_column_repairs=repairs,stop_gradient_verified=True)

def gradients():
    old=torch.get_default_dtype();torch.set_default_dtype(torch.float64);records=[]
    try:
        for name in ['direct','uniform','soft']:
            seed_all(123);g=HeteroData();g['miRNA'].x=torch.randn(5,4);g['drug'].x=torch.randn(4,3)
            g['miRNA','MiMi','miRNA'].edge_index=torch.tensor([[0,0],[1,1],[2,2],[3,3],[4,4],[0,1],[1,0],[2,4],[4,2]]).t()
            g['drug','DrugDrug','drug'].edge_index=torch.tensor([[0,0],[1,1],[2,2],[3,3],[0,1],[1,0]]).t()
            pos=np.array([[0,0],[0,1],[1,0],[1,1],[2,2],[3,1]])
            e=torch.as_tensor(pos.T.copy());g['miRNA','MiDrug','drug'].edge_index=e;g['drug','rev_MiDrug','miRNA'].edge_index=e.flip(0)
            pairs=np.r_[pos,[[4,3],[1,3],[4,0]]];y=np.r_[np.ones(len(pos)),np.zeros(3)]
            data=dict(train_pairs=pairs,train_y=y,val_pairs=pairs,val_y=y,test_pairs=np.empty((0,2),dtype=np.int64))
            def build(graph,record):
                a=configure(Adapter(record,graph,'cpu',ModelConfig(hidden=8,ncn_dim=4)),record,CONFIGS[name]);a.model.eval();return a
            a=build(g,data)
            with torch.no_grad():a.model.soft_paths.head[-1].weight.normal_(std=.1)
            for r,d in [(0,0),(3,1),(4,3),(2,1)]:
                pair=torch.tensor([[r],[d]]);deleted=g.clone()
                for edge,s,t in [(('miRNA','MiDrug','drug'),r,d),(('drug','rev_MiDrug','miRNA'),d,r)]:
                    ei=deleted[edge].edge_index;deleted[edge].edge_index=ei[:,~((ei[0]==s)&(ei[1]==t))]
                keep=~((pairs[:,0]==r)&(pairs[:,1]==d));b=build(deleted,{**data,'train_pairs':pairs[keep],'train_y':y[keep]})
                b.model.load_state_dict(a.model.state_dict())
                def output(obj):return obj.model(pair,obj.ps,(obj.ps.qis(*pair)-a.mu)/a.sd,obj.ps.ease(*pair),obj.dense)
                u,v=output(a),output(b);torch.testing.assert_close(u,v,atol=1e-8,rtol=1e-7)
                ga=torch.autograd.grad(F.softplus(u).sum(),list(a.model.parameters()),allow_unused=True)
                gb=torch.autograd.grad(F.softplus(v).sum(),list(b.model.parameters()),allow_unused=True);err=0.
                for x,z in zip(ga,gb):
                    assert (x is None)==(z is None)
                    if x is not None:torch.testing.assert_close(x,z,atol=2e-7,rtol=2e-6);err=max(err,float(abs(x-z).max()))
                records.append(dict(mode=name,pair=[r,d],max_gradient_error=err,logit_error=float(abs(u-v).max())))
    finally:torch.set_default_dtype(old)
    return records

@torch.no_grad()
def explicit_first(a):
    out={};enc=a.model.encoder
    for edge in a.graph.edge_types:
        conv=enc.original.conv1['__'.join(edge)]
        value=conv((a.graph[edge[0]].x,a.graph[edge[2]].x),a.graph[edge].edge_index)
        out[edge[2]]=out.get(edge[2],0)+value
    return {n:F.leaky_relu(v/enc.counts[n],.1) for n,v in out.items()}

@torch.no_grad()
def feature_check(a,data):
    branch=a.model.soft_paths
    selected=np.r_[np.flatnonzero(data['train_y']==1)[:3],np.flatnonzero(data['train_y']==0)[:3]];records=[]
    for ix in selected:
        r,d=data['train_pairs'][ix];keep=np.arange(len(data['train_y']))!=ix
        other={**data,'train_pairs':data['train_pairs'][keep],'train_y':data['train_y'][keep]}
        name=branch.mode+('_rec10' if hasattr(a.model,'reconstruction') else '')
        b=make_adapter(other,name,a.y.device);b.model.load_state_dict(a.model.state_dict());b.model.eval()
        pair=a.pairs['train'][:,ix:ix+1];direct,fields=branch.features(a.model.encoder,a.ps,a.dense,*pair)
        h=explicit_first(b);s=b.model.soft_paths.scorer;mi,drug=s.mi(h['miRNA']),s.drug(h['drug'])
        expected_direct=s.pairs(mi[pair[0]],drug[pair[1]])
        torch.testing.assert_close(direct,expected_direct,atol=2e-4,rtol=5e-5)
        P=s.matrix(mi,drug).sigmoid().double().cpu().numpy()
        truth=enumeration(b.ps.A.double().cpu().numpy(),b.ps.known.cpu().numpy(),P,r,d,branch.mode)
        got=fields[0].cpu().numpy();np.testing.assert_allclose(got,truth,atol=3e-6,rtol=3e-6)
        records.append(dict(pair=[int(r),int(d)],field_error=float(abs(got-truth).max()),direct_error=float(abs(direct-expected_direct).max())))
    return records

def main():
    operators=operator_checks();grad=gradients();print('PASS operators and final gradients',flush=True)
    data=load_inner(0);seed_all(42);base=make_adapter(data,'control','cuda:0');base.model.eval()
    with torch.no_grad():p0=base.predict('val')
    state={k:v.cpu().clone() for k,v in base.model.state_dict().items()};del base;gc.collect();torch.cuda.empty_cache();rows=[]
    for name,cfg in CONFIGS.items():
        seed_all(42);a=make_adapter(data,name,'cuda:0');m=a.model;m.eval()
        for k,v in state.items():torch.testing.assert_close(m.state_dict()[k].cpu(),v,atol=0,rtol=0)
        with torch.no_grad():p1=a.predict('val');m.soft_paths.head[-1].weight.normal_(std=.1)
        np.testing.assert_allclose(p0,p1,atol=2e-6,rtol=2e-6)
        torch.cuda.reset_peak_memory_stats();start=time.time();m.train();opt=torch.optim.RMSprop(m.parameters(),lr=1e-4);opt.zero_grad()
        loss=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)+cfg['auxiliary_weight']*auxiliary_loss(a)
        if cfg['reconstruction_weight']:
            rec=m.reconstruction;idx=rec.masks(a.graph,.5,mask_generator('cuda:0',0));aux,_=rec.loss(m.encoder,a.graph,idx,False);loss=loss+cfg['reconstruction_weight']*aux
        loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        gn={k:float(p.grad.norm()) for k,p in m.soft_paths.named_parameters()};assert gn['scorer.mi.1.weight']>0 and gn['scorer.drug.1.weight']>0
        opt.step();torch.cuda.synchronize()
        rows.append(dict(config=name,identity_error=float(abs(p0-p1).max()),step_seconds=time.time()-start,
            peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,branch_gradients=gn,
            enumerated_features=feature_check(a,data),explicit_deletions=deletion_check(a,data,name)))
        print('PASS',name,flush=True);del a,m,opt;gc.collect();torch.cuda.empty_cache()
    a=make_adapter(data,'soft','cuda:0');a.model.eval()
    with torch.no_grad():a.model.soft_paths.head[-1].weight.normal_(std=.1)
    b=make_adapter({**data,'val_y':1-data['val_y']},'soft','cuda:0');b.model.load_state_dict(a.model.state_dict());b.model.eval()
    torch.testing.assert_close(a.ps.known,b.ps.known,atol=0,rtol=0)
    with torch.no_grad():pa,pb=a.predict('val'),b.predict('val')
    np.testing.assert_allclose(pa,pb,atol=2e-6,rtol=2e-6)
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',operator_checks=operators,gradient_checks=grad,records=rows,
        heldout_flip_error=float(abs(pa-pb).max()),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
