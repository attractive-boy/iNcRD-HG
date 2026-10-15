"""Path enumeration, gradient deletion, coupling contrast and real-data memory."""
import gc,json,time
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from common import HERE,DEGREE,Adapter,read_module,load_inner,seed_all
from model import CONFIGS,configure,make_adapter,mask_generator,moments
from incrd_hg_m123.config import ModelConfig

QUERY_CHECK_SOURCE=DEGREE/'check.py'
checker=read_module('path_moment_deletion_helper',QUERY_CHECK_SOURCE);checker.make_adapter=make_adapter
deletion_check=checker.deletion_check


def enumeration(A,r,d,phi,psi):
    output_dtype=phi.dtype;phi,psi=phi.double(),psi.double()
    ii,jj=torch.where(A>0);take=(ii!=r)&(jj!=d)&(A[r,jj]>0)&(A[ii,d]>0);ii,jj=ii[take],jj[take]
    count=len(ii)
    if not count:return {k:phi.new_zeros(phi.shape[1],dtype=output_dtype) for k in ['independent','joint','covariance']},0
    left,right=phi[ii],psi[jj];joint=(left*right).mean(0);independent=left.mean(0)*right.mean(0)
    covariance=((left-left.mean(0))*(right-right.mean(0))).mean(0)
    return {k:(count/(count+1)*v).to(output_dtype) for k,v in [('independent',independent),('joint',joint),('covariance',covariance)]},count


def coupling_example():
    phi=torch.tensor([[0.],[-1.],[1.]]);psi=phi.clone();zero=torch.zeros((1,1));S=torch.zeros(3,3)
    values=[]
    for edges in [[(1,1),(2,2)],[(1,2),(2,1)]]:
        A=torch.zeros(3,3);A[0,1:]=1;A[1:,0]=1
        for i,j in edges:A[i,j]=1
        out,count=moments(A,S,S,torch.tensor([0]),torch.tensor([0]),phi,psi,zero,zero,zero,zero)
        for key,value in out.items():torch.testing.assert_close(value[0],enumeration(A,0,0,phi,psi)[0][key])
        values.append({k:float(v) for k,v in out.items()})
    assert values[0]['independent']==values[1]['independent']==0 and values[0]['joint']>0>values[1]['joint']
    return dict(same_marginal_means=True,diagonal=values[0],crossed=values[1],scope='summary with fixed intermediate states, not an expressivity proof for the entire original network')


def gradient_checks():
    old=torch.get_default_dtype();torch.set_default_dtype(torch.float64);records=[]
    try:
        for name,cfg in CONFIGS.items():
            if cfg['reconstruction_weight']:continue
            seed_all(123);g=HeteroData();g['miRNA'].x=torch.randn(5,4);g['drug'].x=torch.randn(4,3)
            g['miRNA','MiMi','miRNA'].edge_index=torch.tensor([[0,0],[0,0],[1,1],[2,2],[3,3],[4,4],[0,1],[1,0],[2,4],[4,2]]).t()
            g['drug','DrugDrug','drug'].edge_index=torch.tensor([[0,0],[1,1],[2,2],[3,3],[2,2],[0,1],[1,0]]).t()
            pos=np.array([[0,0],[0,1],[1,0],[1,1],[2,2],[3,1]])
            e=torch.as_tensor(pos.T.copy());g['miRNA','MiDrug','drug'].edge_index=e;g['drug','rev_MiDrug','miRNA'].edge_index=e.flip(0)
            pairs=np.r_[pos,[[4,3],[1,3],[4,0]]];y=np.r_[np.ones(len(pos)),np.zeros(3)]
            data=dict(train_pairs=pairs,train_y=y,val_pairs=pairs,val_y=y,test_pairs=np.empty((0,2),dtype=np.int64))
            def build(graph,record):
                a=configure(Adapter(record,graph,'cpu',ModelConfig(hidden=8,ncn_dim=4)),cfg);a.model.eval();return a
            a=build(g,data)
            with torch.no_grad():a.model.path_moment.head[-1].weight.normal_(std=.1)
            for r,d in [(0,0),(3,1),(4,3),(2,1)]:
                pair=torch.tensor([[r],[d]]);deleted=g.clone()
                for edge,s,t in [(('miRNA','MiDrug','drug'),r,d),(('drug','rev_MiDrug','miRNA'),d,r)]:
                    ei=deleted[edge].edge_index;deleted[edge].edge_index=ei[:,~((ei[0]==s)&(ei[1]==t))]
                keep=~((pairs[:,0]==r)&(pairs[:,1]==d));other={**data,'train_pairs':pairs[keep],'train_y':y[keep]}
                b=build(deleted,other);b.model.load_state_dict(a.model.state_dict())
                def scalar(obj):
                    qi=(obj.ps.qis(*pair)-a.mu)/a.sd;out=obj.model(pair,obj.ps,qi,obj.ps.ease(*pair),obj.dense)
                    z=obj.model.encoder.forward_dense(*obj.dense,*pair,obj.ps.qvals(*pair))
                    feat,count=obj.model.path_moment.features(obj.ps,*pair,z[0],z[1],z[4])
                    return F.softplus(out).sum()+.001*sum(v.square().sum() for v in feat.values()),out,feat,count
                l1,s1,f1,count=scalar(a);l2,s2,f2,count2=scalar(b)
                torch.testing.assert_close(s1,s2,atol=1e-8,rtol=1e-7);torch.testing.assert_close(l1,l2,atol=1e-8,rtol=1e-7)
                with torch.no_grad():
                    z=b.model.encoder(b.graph.x_dict,b.graph.edge_index_dict)
                    truth,n=enumeration(b.ps.A,r,d,b.model.path_moment.rproj(z['miRNA']),b.model.path_moment.dproj(z['drug']))
                    assert float(count)==float(count2)==n
                    for k in truth:torch.testing.assert_close(f1[k][0],truth[k],atol=1e-8,rtol=1e-7)
                ga=torch.autograd.grad(l1,list(a.model.parameters()),allow_unused=True)
                gb=torch.autograd.grad(l2,list(b.model.parameters()),allow_unused=True);err=0.
                for u,v in zip(ga,gb):
                    assert (u is None)==(v is None)
                    if u is not None:torch.testing.assert_close(u,v,atol=2e-7,rtol=2e-6);err=max(err,float(abs(u-v).max()))
                records.append(dict(config=name,pair=[r,d],paths=n,max_gradient_error=err,score_error=float(abs(s1-s2).max())))
    finally:torch.set_default_dtype(old)
    return records


@torch.no_grad()
def feature_deletion_check(a,data,name):
    a.model.eval();selected=np.r_[np.flatnonzero(data['train_y']==1)[:3],np.flatnonzero(data['train_y']==0)[:3]];rows=[]
    for ix in selected:
        keep=np.arange(len(data['train_y']))!=ix;other={**data,'train_pairs':data['train_pairs'][keep],'train_y':data['train_y'][keep]}
        b=make_adapter(other,name,a.y.device);b.model.load_state_dict(a.model.state_dict());b.model.eval()
        pair=a.pairs['train'][:,ix:ix+1];r,d=pair
        z=a.model.encoder.forward_dense(*a.dense,r,d,a.ps.qvals(r,d));feat,count=a.model.path_moment.features(a.ps,r,d,z[0],z[1],z[4])
        explicit=b.model.encoder(b.graph.x_dict,b.graph.edge_index_dict)
        truth,n=enumeration(b.ps.A,int(r),int(d),b.model.path_moment.rproj(explicit['miRNA']),b.model.path_moment.dproj(explicit['drug']))
        errors={}
        assert float(count)==n
        for k,v in truth.items():
            torch.testing.assert_close(feat[k][0],v,atol=3e-4,rtol=1e-4);errors[k]=float(abs(feat[k][0]-v).max())
        rows.append(dict(pair=data['train_pairs'][ix].tolist(),paths=n,errors=errors))
    return rows


def main():
    example=coupling_example();gradients=gradient_checks();print('PASS coupling and gradients',len(gradients),flush=True)
    data=load_inner(0);seed_all(42);base=make_adapter(data,'control','cuda:0');base.model.eval()
    with torch.no_grad():p0=base.predict('val')
    state={k:v.cpu().clone() for k,v in base.model.state_dict().items()};del base;gc.collect();torch.cuda.empty_cache();rows=[]
    for name,cfg in CONFIGS.items():
        seed_all(42);a=make_adapter(data,name,'cuda:0');m=a.model;m.eval()
        for k,v in state.items():torch.testing.assert_close(m.state_dict()[k].cpu(),v,atol=0,rtol=0)
        with torch.no_grad():p1=a.predict('val');m.path_moment.head[-1].weight.normal_(std=.01)
        np.testing.assert_allclose(p0,p1,atol=2e-6,rtol=2e-6)
        torch.cuda.reset_peak_memory_stats();start=time.time();m.train();opt=torch.optim.RMSprop(m.parameters(),lr=1e-4);opt.zero_grad()
        loss=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)
        if cfg['reconstruction_weight']:
            rec=m.reconstruction;ix=rec.masks(a.graph,.5,mask_generator('cuda:0',0));aux,_=rec.loss(m.encoder,a.graph,ix,False);loss=loss+cfg['reconstruction_weight']*aux
        loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        gn={k:float(p.grad.norm()) for k,p in m.path_moment.named_parameters() if p.grad is not None}
        assert gn['rproj.weight']>0 and gn['dproj.weight']>0;opt.step();torch.cuda.synchronize()
        peak=torch.cuda.max_memory_allocated()/2**30;elapsed=time.time()-start
        rows.append(dict(config=name,identity_max_prediction_error=float(abs(p0-p1).max()),peak_cuda_gib=peak,step_seconds=elapsed,
            branch_gradient_norms=gn,explicit_deletions=deletion_check(a,data,name),enumerated_features=feature_deletion_check(a,data,name)))
        print('PASS',name,peak,elapsed,flush=True);del a,m,opt;gc.collect();torch.cuda.empty_cache()
    seed_all(42);a=make_adapter(data,'joint','cuda:0');a.model.eval()
    with torch.no_grad():a.model.path_moment.head[-1].weight.normal_(std=.1)
    b=make_adapter({**data,'val_y':1-data['val_y']},'joint','cuda:0');b.model.load_state_dict(a.model.state_dict());b.model.eval()
    with torch.no_grad():pa,pb=a.predict('val'),b.predict('val')
    np.testing.assert_allclose(pa,pb,atol=2e-6,rtol=2e-6)
    result=dict(status='passed',coupling_example=example,gradient_checks=gradients,records=rows,heldout_flip_error=float(abs(pa-pb).max()),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'initial_checks.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
