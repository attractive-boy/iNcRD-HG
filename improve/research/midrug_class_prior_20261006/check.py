"""Check independent count reconstruction, query deletion and final gradients."""
import json,time,gc
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from common import HERE,DEGREE,read_module,Adapter,load_inner,seed_all
from model import CONFIGS,GroupStats,make_adapter,configure,mask_generator
from incrd_hg_m123.config import ModelConfig

QUERY_CHECK_SOURCE=DEGREE/'check.py'
checker=read_module('class_prior_deletion_helper',QUERY_CHECK_SOURCE);checker.make_adapter=make_adapter
deletion_check=checker.deletion_check

def direct(A,mi,drug,r,d):
    """Independently enumerate edges after physically removing the query."""
    edges=[(i,j) for i,j in zip(*np.where(A>0)) if (i,j)!=(r,d)]
    block=sum(mi[i]==mi[r] and drug[j]==drug[d] for i,j in edges)
    row=sum(mi[i]==mi[r] for i,j in edges);col=sum(drug[j]==drug[d] for i,j in edges)
    pr=sum(mi==mi[r])/len(mi);pd=sum(drug==drug[d])/len(drug)
    return np.log([(block+10*pd)/(row+10)/pd,(block+10*pr)/(col+10)/pr])

def count_checks():
    A=torch.tensor([[1.,1.,0.,0.],[0.,1.,1.,0.],[0.,0.,0.,0.],[0.,0.,0.,1.]],dtype=torch.float64)
    mi=np.array([0,0,1,2]);drug=np.array([0,1,0,1]);stats=GroupStats(A,mi,drug);records=[]
    for r in range(4):
        for d in range(4):
            idx=(torch.tensor([r]),torch.tensor([d]));got=stats.features(*idx)[0].numpy()
            expected=direct(A.numpy(),mi,drug,r,d);np.testing.assert_allclose(got,expected,atol=1e-12,rtol=1e-12)
            B=A.clone();B[r,d]=0;other=GroupStats(B,mi,drug)
            torch.testing.assert_close(stats.features(*idx),other.features(*idx),atol=0,rtol=0)
            records.append(dict(pair=[r,d],positive=bool(A[r,d]),error=float(abs(got-expected).max())))
    zero=GroupStats(torch.zeros_like(A),mi,drug).features(torch.arange(4),torch.arange(4))
    torch.testing.assert_close(zero,torch.zeros_like(zero),atol=1e-15,rtol=0)
    return records

def gradients():
    old=torch.get_default_dtype();torch.set_default_dtype(torch.float64);records=[]
    try:
        seed_all(123);g=HeteroData();g['miRNA'].x=torch.randn(5,4);g['drug'].x=torch.randn(4,3)
        g['miRNA','MiMi','miRNA'].edge_index=torch.tensor([[0,0],[1,1],[2,2],[3,3],[4,4],[0,1],[1,0],[2,4],[4,2]]).t()
        g['drug','DrugDrug','drug'].edge_index=torch.tensor([[0,0],[1,1],[2,2],[3,3],[0,1],[1,0]]).t()
        pos=np.array([[0,0],[0,1],[1,0],[1,1],[2,2],[3,1]])
        e=torch.as_tensor(pos.T.copy());g['miRNA','MiDrug','drug'].edge_index=e;g['drug','rev_MiDrug','miRNA'].edge_index=e.flip(0)
        pairs=np.r_[pos,[[4,3],[1,3],[4,0]]];y=np.r_[np.ones(len(pos)),np.zeros(3)]
        data=dict(train_pairs=pairs,train_y=y,val_pairs=pairs,val_y=y,test_pairs=np.empty((0,2),dtype=np.int64))
        groups=(np.array([0,0,1,1,2]),np.array([0,0,1,1]))
        def build(graph,record):
            a=configure(Adapter(record,graph,'cpu',ModelConfig(hidden=8,ncn_dim=4)),CONFIGS['feature'],groups);a.model.eval();return a
        a=build(g,data)
        with torch.no_grad():a.model.class_prior.net[-1].weight.normal_(std=.1)
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
            records.append(dict(pair=[r,d],max_gradient_error=err,logit_error=float(abs(u-v).max())))
    finally:torch.set_default_dtype(old)
    return records

@torch.no_grad()
def feature_check(a,data):
    stats=a.ps.class_stats;selected=np.r_[np.flatnonzero(data['train_y']==1)[:3],np.flatnonzero(data['train_y']==0)[:3]]
    A=stats.A.cpu().numpy();mi=stats.mi.cpu().numpy();drug=stats.drug.cpu().numpy();records=[]
    for ix in selected:
        r,d=data['train_pairs'][ix];got=stats.features(*a.pairs['train'][:,ix:ix+1])[0].cpu().numpy()
        expected=direct(A,mi,drug,r,d);np.testing.assert_allclose(got,expected,atol=1e-6,rtol=1e-6)
        records.append(dict(pair=[int(r),int(d)],error=float(abs(got-expected).max())))
    return records

def main():
    counts=count_checks();grad=gradients();print('PASS counts and final gradients',flush=True)
    data=load_inner(0);seed_all(42);base=make_adapter(data,'control','cuda:0');base.model.eval()
    with torch.no_grad():p0=base.predict('val')
    state={k:v.cpu().clone() for k,v in base.model.state_dict().items()};del base;gc.collect();torch.cuda.empty_cache();rows=[]
    for name,cfg in CONFIGS.items():
        seed_all(42);a=make_adapter(data,name,'cuda:0');m=a.model;m.eval()
        for k,v in state.items():torch.testing.assert_close(m.state_dict()[k].cpu(),v,atol=0,rtol=0)
        with torch.no_grad():p1=a.predict('val');m.class_prior.net[-1].weight.normal_(std=.1)
        np.testing.assert_allclose(p0,p1,atol=2e-6,rtol=2e-6)
        torch.cuda.reset_peak_memory_stats();start=time.time();m.train();opt=torch.optim.RMSprop(m.parameters(),lr=1e-4);opt.zero_grad()
        loss=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)
        if cfg['reconstruction_weight']:
            rec=m.reconstruction;idx=rec.masks(a.graph,.5,mask_generator('cuda:0',0));aux,_=rec.loss(m.encoder,a.graph,idx,False);loss=loss+cfg['reconstruction_weight']*aux
        loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        gn={k:float(p.grad.norm()) for k,p in m.class_prior.named_parameters()};assert all(v>0 for v in gn.values())
        opt.step();torch.cuda.synchronize()
        rows.append(dict(config=name,identity_error=float(abs(p0-p1).max()),step_seconds=time.time()-start,
            peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,branch_gradients=gn,
            enumerated_features=feature_check(a,data),explicit_deletions=deletion_check(a,data,name)))
        print('PASS',name,flush=True);del a,m,opt;gc.collect();torch.cuda.empty_cache()
    a=make_adapter(data,'feature','cuda:0');a.model.eval()
    with torch.no_grad():a.model.class_prior.net[-1].weight.normal_(std=.1)
    b=make_adapter({**data,'val_y':1-data['val_y']},'feature','cuda:0');b.model.load_state_dict(a.model.state_dict());b.model.eval()
    torch.testing.assert_close(a.ps.class_stats.table,b.ps.class_stats.table,atol=0,rtol=0)
    with torch.no_grad():pa,pb=a.predict('val'),b.predict('val')
    np.testing.assert_allclose(pa,pb,atol=2e-6,rtol=2e-6)
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',count_checks=counts,gradient_checks=grad,records=rows,
        heldout_flip_error=float(abs(pa-pb).max()),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
