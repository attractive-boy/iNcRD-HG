"""Feature permutation, heldout isolation and explicit-deletion gradient checks."""
import gc,json,time
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from common import HERE,DEGREE,Adapter,read_module,load_inner,seed_all
from model import CONFIGS,GlobalFeatures,GlobalFeatureEncoder,make_adapter,mask_generator
from incrd_hg_m123.config import ModelConfig

QUERY_CHECK_SOURCE=DEGREE/'check.py'
checker=read_module('global_features_real_deletion',QUERY_CHECK_SOURCE)
checker.make_adapter=make_adapter
deletion_check=checker.deletion_check


def gradient_checks():
    old=torch.get_default_dtype();torch.set_default_dtype(torch.float64);records=[]
    try:
        for mode in ('local','pool','route'):
            seed_all(42);g=HeteroData();g['miRNA'].x=torch.randn(5,4);g['drug'].x=torch.randn(4,3)
            g['miRNA','MiMi','miRNA'].edge_index=torch.tensor([[0,0],[0,0],[1,1],[2,2],[3,3],[4,4],[0,1],[1,0]]).t()
            g['drug','DrugDrug','drug'].edge_index=torch.tensor([[0,0],[1,1],[2,2],[3,3],[2,2],[0,1],[1,0]]).t()
            pos=np.array([[0,0],[0,1],[1,0],[2,2],[3,1]])
            e=torch.as_tensor(pos.T.copy());g['miRNA','MiDrug','drug'].edge_index=e;g['drug','rev_MiDrug','miRNA'].edge_index=e.flip(0)
            pairs=np.r_[pos,[[4,3],[1,3],[4,0]]];y=np.r_[np.ones(5),np.zeros(3)]
            data=dict(train_pairs=pairs,train_y=y,val_pairs=pairs,val_y=y,test_pairs=np.empty((0,2),dtype=np.int64))
            def build(graph,record):
                a=Adapter(record,graph,'cpu',ModelConfig(hidden=8,ncn_dim=4))
                a.model.encoder.__class__=GlobalFeatureEncoder
                a.model.encoder.add_module('refiner',GlobalFeatures(graph,mode,width=8,routes=3))
                a.model.eval();return a
            a=build(g,data)
            with torch.no_grad():
                for module in a.model.encoder.refiner.output.values():module[-1].weight.normal_(std=.1)
            for r,d in [(0,0),(3,1),(4,3)]:
                pair=torch.tensor([[r],[d]]);deleted=g.clone()
                for edge,s,t in [(('miRNA','MiDrug','drug'),r,d),(('drug','rev_MiDrug','miRNA'),d,r)]:
                    ei=deleted[edge].edge_index;deleted[edge].edge_index=ei[:,~((ei[0]==s)&(ei[1]==t))]
                keep=~((pairs[:,0]==r)&(pairs[:,1]==d));other={**data,'train_pairs':pairs[keep],'train_y':y[keep]}
                b=build(deleted,other);b.model.load_state_dict(a.model.state_dict())
                def scalar(obj):
                    qi=(obj.ps.qis(*pair)-a.mu)/a.sd
                    out=obj.model(pair,obj.ps,qi,obj.ps.ease(*pair),obj.dense)
                    z=obj.model.encoder.forward_dense(*obj.dense,*pair,obj.ps.qvals(*pair))
                    pools=obj.ps.qncn(*pair,{'miRNA':z[0],'drug':z[1]},z[4])
                    return F.softplus(out).sum()+.01*sum(v.square().sum() for v in [z[2],z[3],*pools]),out
                l1,s1=scalar(a);l2,s2=scalar(b)
                torch.testing.assert_close(s1,s2,atol=1e-9,rtol=1e-8)
                ga=torch.autograd.grad(l1,[p for p in a.model.parameters() if p.requires_grad],allow_unused=True)
                gb=torch.autograd.grad(l2,[p for p in b.model.parameters() if p.requires_grad],allow_unused=True)
                error=0.
                for u,v in zip(ga,gb):
                    assert (u is None)==(v is None)
                    if u is not None:
                        torch.testing.assert_close(u,v,atol=2e-8,rtol=2e-7);error=max(error,float(abs(u-v).max()))
                records.append(dict(mode=mode,pair=[r,d],score_error=float(abs(s1-s2).max()),max_gradient_error=error))
    finally:torch.set_default_dtype(old)
    return records


def main():
    gradients=gradient_checks();print('PASS gradient comparisons',len(gradients),flush=True)
    data=load_inner(0);seed_all(42);base=make_adapter(data,'control','cuda:0');base.model.eval()
    with torch.no_grad():p0=base.predict('val')
    base_state={k:v.cpu().clone() for k,v in base.model.state_dict().items()};del base;gc.collect();torch.cuda.empty_cache()
    rows=[]
    for name,cfg in CONFIGS.items():
        seed_all(42);a=make_adapter(data,name,'cuda:0');m=a.model;m.eval()
        for k,v in base_state.items():torch.testing.assert_close(m.state_dict()[k].cpu(),v,atol=0,rtol=0)
        with torch.no_grad():p1=a.predict('val')
        np.testing.assert_allclose(p0,p1,atol=1e-6,rtol=1e-6)
        ref=m.encoder.refiner
        with torch.no_grad():
            for module in ref.output.values():module[-1].weight.normal_(std=.01)
            x=a.graph.x_dict;original={k:v.clone() for k,v in x.items()};transformed=ref(x)
            perm={k:torch.randperm(len(v),device=v.device) for k,v in x.items()}
            shuffled=ref({k:v[perm[k]] for k,v in x.items()})
            perr=max(float(abs(shuffled[k]-transformed[k][perm[k]]).max()) for k in x)
            for k in x:
                torch.testing.assert_close(shuffled[k],transformed[k][perm[k]],atol=2e-6,rtol=2e-6)
                torch.testing.assert_close(x[k],original[k],atol=0,rtol=0)
        torch.cuda.reset_peak_memory_stats();start=time.time();m.train()
        opt=torch.optim.RMSprop([p for p in m.parameters() if p.requires_grad],lr=1e-4);opt.zero_grad()
        bce=F.binary_cross_entropy_with_logits(a.logits('train'),a.y);loss=bce
        if cfg['reconstruction_weight']:
            rec=m.reconstruction;ix=rec.masks(a.graph,.5,mask_generator('cuda:0',0));aux,_=rec.loss(m.encoder,a.graph,ix,False)
            loss=loss+cfg['reconstruction_weight']*aux
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        active={k:float(p.grad.norm()) for k,p in ref.named_parameters() if p.requires_grad and p.grad is not None}
        if cfg['mode']=='route':assert all(any(k.startswith(prefix) and v>0 for k,v in active.items()) for prefix in ['key.','query.','tokens.','value.'])
        opt.step();torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()/2**30;elapsed=time.time()-start
        deletion=deletion_check(a,data,name)
        rows.append(dict(config=name,identity_max_prediction_error=float(abs(p0-p1).max()),permutation_max_error=perr,
            peak_cuda_gib=peak,step_seconds=elapsed,refiner_gradient_norms=active,explicit_deletions=deletion))
        print('PASS',name,flush=True);del a,m,ref,opt;gc.collect();torch.cuda.empty_cache()
    seed_all(42);a=make_adapter(data,'route','cuda:0');a.model.eval()
    b=make_adapter({**data,'val_y':1-data['val_y']},'route','cuda:0');b.model.load_state_dict(a.model.state_dict());b.model.eval()
    with torch.no_grad():pa,pb=a.predict('val'),b.predict('val')
    np.testing.assert_allclose(pa,pb,atol=1e-6,rtol=1e-6)
    report=dict(status='passed',gradient_checks=gradients,records=rows,heldout_flip_error=float(abs(pa-pb).max()),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'initial_checks.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
