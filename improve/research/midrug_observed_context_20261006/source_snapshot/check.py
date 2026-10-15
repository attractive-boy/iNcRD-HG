"""Positive/zero/nonedge deletion, final-score gradients and label isolation."""
import gc,json,time
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from common import HERE,DEGREE,Adapter,read_module,load_inner,seed_all
from model import CONFIGS,configure,make_adapter,mask_generator
from incrd_hg_m123.config import ModelConfig

QUERY_CHECK_SOURCE=DEGREE/'check.py'
checker=read_module('observed_context_deletion_helper',QUERY_CHECK_SOURCE)
checker.make_adapter=make_adapter
deletion_check=checker.deletion_check


def gradient_checks():
    old=torch.get_default_dtype();torch.set_default_dtype(torch.float64);records=[]
    try:
        for name,cfg in CONFIGS.items():
            if cfg['reconstruction_weight']:continue
            seed_all(123);g=HeteroData();g['miRNA'].x=torch.randn(5,4);g['drug'].x=torch.randn(4,3)
            g['miRNA','MiMi','miRNA'].edge_index=torch.tensor([[0,0],[0,0],[1,1],[2,2],[3,3],[4,4],[0,1],[1,0],[2,4],[4,2]]).t()
            g['drug','DrugDrug','drug'].edge_index=torch.tensor([[0,0],[1,1],[2,2],[3,3],[2,2],[0,1],[1,0]]).t()
            pos=np.array([[0,0],[0,1],[1,0],[2,2],[3,1]])
            e=torch.as_tensor(pos.T.copy());g['miRNA','MiDrug','drug'].edge_index=e;g['drug','rev_MiDrug','miRNA'].edge_index=e.flip(0)
            pairs=np.r_[pos,[[4,3],[1,3],[4,0]]];y=np.r_[np.ones(5),np.zeros(3)]
            data=dict(train_pairs=pairs,train_y=y,val_pairs=pairs,val_y=y,test_pairs=np.empty((0,2),dtype=np.int64))
            def build(graph,record):
                a=configure(Adapter(record,graph,'cpu',ModelConfig(hidden=8,ncn_dim=4)),record,cfg)
                a.model.eval();return a
            a=build(g,data)
            with torch.no_grad():
                for layer in [a.model.encoder.extra1,a.model.encoder.extra2]:
                    for module in layer.values():module.weight.normal_(std=.1)
            for r,d in [(0,0),(3,1),(1,3),(4,3),(2,1)]:
                pair=torch.tensor([[r],[d]]);deleted=g.clone()
                for edge,s,t in [(('miRNA','MiDrug','drug'),r,d),(('drug','rev_MiDrug','miRNA'),d,r)]:
                    ei=deleted[edge].edge_index;deleted[edge].edge_index=ei[:,~((ei[0]==s)&(ei[1]==t))]
                keep=~((pairs[:,0]==r)&(pairs[:,1]==d));other={**data,'train_pairs':pairs[keep],'train_y':y[keep]}
                b=build(deleted,other);b.model.load_state_dict(a.model.state_dict())
                assert b.model.encoder.context[r,d]==0 and b.ps.A[r,d]==0
                def scalar(obj):
                    qi=(obj.ps.qis(*pair)-a.mu)/a.sd
                    out=obj.model(pair,obj.ps,qi,obj.ps.ease(*pair),obj.dense)
                    z=obj.model.encoder.forward_dense(*obj.dense,*pair,obj.ps.qvals(*pair))
                    pools=obj.ps.qncn(*pair,{'miRNA':z[0],'drug':z[1]},z[4])
                    return F.softplus(out).sum()+.01*sum(v.square().sum() for v in [z[2],z[3],*pools]),out
                l1,s1=scalar(a);l2,s2=scalar(b)
                torch.testing.assert_close(s1,s2,atol=1e-9,rtol=1e-8);torch.testing.assert_close(l1,l2,atol=1e-9,rtol=1e-8)
                ga=torch.autograd.grad(l1,[p for p in a.model.parameters() if p.requires_grad],allow_unused=True)
                gb=torch.autograd.grad(l2,[p for p in b.model.parameters() if p.requires_grad],allow_unused=True)
                error=0.
                for u,v in zip(ga,gb):
                    assert (u is None)==(v is None)
                    if u is not None:
                        torch.testing.assert_close(u,v,atol=2e-8,rtol=2e-7);error=max(error,float(abs(u-v).max()))
                records.append(dict(config=name,pair=[r,d],positive_membership=float(a.ps.A[r,d]),
                    context_membership=float(a.model.encoder.context[r,d]),score_error=float(abs(s1-s2).max()),max_gradient_error=error))
    finally:torch.set_default_dtype(old)
    return records


def main():
    gradients=gradient_checks();print('PASS gradient comparisons',len(gradients),flush=True)
    data=load_inner(0);seed_all(42);base=make_adapter(data,'control','cuda:0');base.model.eval()
    with torch.no_grad():p0=base.predict('val')
    state={k:v.cpu().clone() for k,v in base.model.state_dict().items()};del base;gc.collect();torch.cuda.empty_cache()
    rows=[]
    for name,cfg in CONFIGS.items():
        seed_all(42);a=make_adapter(data,name,'cuda:0');m=a.model;m.eval()
        for k,v in state.items():torch.testing.assert_close(m.state_dict()[k].cpu(),v,atol=0,rtol=0)
        assert int(m.encoder.context.sum())==int((data['train_y']==cfg['support']).sum())
        assert not any(k.endswith('context') for k in m.state_dict())
        with torch.no_grad():p1=a.predict('val')
        np.testing.assert_allclose(p0,p1,atol=2e-6,rtol=2e-6)
        with torch.no_grad():
            for layer in [m.encoder.extra1,m.encoder.extra2]:
                for module in layer.values():module.weight.normal_(std=.01)
        torch.cuda.reset_peak_memory_stats();start=time.time();m.train()
        opt=torch.optim.RMSprop(m.parameters(),lr=1e-4);opt.zero_grad()
        loss=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)
        if cfg['reconstruction_weight']:
            rec=m.reconstruction;ix=rec.masks(a.graph,.5,mask_generator('cuda:0',0));aux,_=rec.loss(m.encoder,a.graph,ix,False)
            loss=loss+cfg['reconstruction_weight']*aux
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        active={k:float(p.grad.norm()) for k,p in m.named_parameters() if k.startswith('encoder.extra')}
        assert all(v>0 for v in active.values());opt.step();torch.cuda.synchronize()
        peak=torch.cuda.max_memory_allocated()/2**30;elapsed=time.time()-start
        deletion=deletion_check(a,data,name)
        rows.append(dict(config=name,identity_max_prediction_error=float(abs(p0-p1).max()),peak_cuda_gib=peak,
            step_seconds=elapsed,context_edges=int(m.encoder.context.sum()),extra_gradient_norms=active,explicit_deletions=deletion))
        print('PASS',name,flush=True);del a,m,opt;gc.collect();torch.cuda.empty_cache()
    seed_all(42);a=make_adapter(data,'zero_mean','cuda:0');a.model.eval()
    with torch.no_grad():
        for layer in [a.model.encoder.extra1,a.model.encoder.extra2]:
            for module in layer.values():module.weight.normal_(std=.03)
    b=make_adapter({**data,'val_y':1-data['val_y']},'zero_mean','cuda:0');b.model.load_state_dict(a.model.state_dict());b.model.eval()
    torch.testing.assert_close(a.model.encoder.context,b.model.encoder.context,atol=0,rtol=0)
    with torch.no_grad():pa,pb=a.predict('val'),b.predict('val')
    np.testing.assert_allclose(pa,pb,atol=1e-6,rtol=1e-6)
    result=dict(status='passed',gradient_checks=gradients,records=rows,heldout_flip_error=float(abs(pa-pb).max()),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'initial_checks.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
