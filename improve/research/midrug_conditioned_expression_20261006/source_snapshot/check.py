"""Dense single-context physical deletion oracle for the conditioned branch."""
import json,time
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,DEGREE,read_module,load_inner,seed_all,sha256
from model import CONFIGS,ConditionedExpression,make_adapter,make_optimizer,mask_generator
QUERY_CHECK_SOURCE=DEGREE/'check.py'
_query=read_module('conditioned_expression_full_query_checks',QUERY_CHECK_SOURCE);_query.make_adapter=make_adapter
deletion_check=_query.deletion_check

def independent_deleted(m,xr,xd,A,Sr,Sd,r,d):
    A=A.clone();A[r,d]=0
    hr,hd=m.rinput(xr),m.dinput(xd)
    cond=hd[d] if m.conditioned else hd.mean(0)
    gamma,beta=m.modulator(cond).chunk(2,dim=0)
    hm=F.leaky_relu((1+torch.tanh(gamma))*hr+beta,.1)
    for layer in range(2):
        pm=.5*(A@m.dm[layer](hd)+Sr@m.mm[layer](hm))+m.sm[layer](hm)
        pd=.5*(A.T@m.md[layer](hm)+Sd@m.dd[layer](hd))+m.sd[layer](hd)
        hm=F.leaky_relu(F.layer_norm(pm,(m.hidden,),m.nm[layer].weight,m.nm[layer].bias,m.nm[layer].eps),.1)
        hd=F.leaky_relu(F.layer_norm(pd,(m.hidden,),m.nd[layer].weight,m.nd[layer].bias,m.nd[layer].eps),.1)
    return torch.cat([hm[r],hd[d]])

def toy_checks():
    A=torch.tensor([[1.,1.,0.,0.],[0.,1.,1.,0.],[2.,0.,1.,0.],[0.,0.,0.,1.]],dtype=torch.double)
    Sr=torch.tensor([[2.,2.,0.,0.],[1.,1.,1.,0.],[0.,0.,0.,0.],[0.,0.,0.,1.]],dtype=torch.double)
    Sd=torch.tensor([[1.,1.,0.,0.],[0.,2.,1.,0.],[1.,0.,0.,0.],[0.,0.,0.,1.]],dtype=torch.double)
    records=[]
    for cond in [True,False]:
        seed_all(19);xr=torch.randn(4,7,dtype=torch.double);xd=torch.randn(4,6,dtype=torch.double)
        m=ConditionedExpression(7,6,5,cond).double()
        with torch.no_grad():m.modulator[-1].weight.normal_(std=.1);m.modulator[-1].bias.normal_(std=.1)
        params=list(m.parameters())
        for r,d in [(0,0),(2,0),(3,3),(0,2),(3,0)]:
            z=m(xr,xd,A,Sr,Sd,torch.tensor([r]),torch.tensor([d]))[0];ref=independent_deleted(m,xr,xd,A,Sr,Sd,r,d)
            torch.testing.assert_close(z,ref,atol=1e-10,rtol=1e-10)
            w=torch.linspace(-.7,.9,len(z),dtype=z.dtype)
            ga=torch.autograd.grad((w*z).sum(),params,allow_unused=True);gb=torch.autograd.grad((w*ref).sum(),params,allow_unused=True)
            errors=[]
            for a,b in zip(ga,gb):
                if a is None or b is None:assert (a is None or torch.count_nonzero(a)==0) and (b is None or torch.count_nonzero(b)==0)
                else:torch.testing.assert_close(a,b,atol=2e-9,rtol=2e-9);errors.append(float(abs(a-b).max()))
            records.append(dict(conditioned=cond,pair=[r,d],removed_weight=float(A[r,d]),output_error=float(abs(z-ref).max()),gradient_error=max(errors)))
    return records

@torch.no_grad()
def real_branch_check(a,data,count=3):
    m=a.model;ix=np.r_[np.flatnonzero(data['train_y']==1)[:count],np.flatnonzero(data['train_y']==0)[:2]];records=[]
    for i in ix:
        r,d=map(int,data['train_pairs'][i]);rt=torch.tensor([r],device=a.y.device);dt=torch.tensor([d],device=a.y.device)
        z=m.expression_context(a.dense[0],a.dense[1],a.ps.A,m.context_sr,m.context_sd,rt,dt)[0]
        ref=independent_deleted(m.expression_context,a.dense[0],a.dense[1],a.ps.A,m.context_sr,m.context_sd,r,d)
        torch.testing.assert_close(z,ref,atol=2e-4,rtol=5e-5)
        records.append(dict(pair=[r,d],label=int(data['train_y'][i]),max_error=float(abs(z-ref).max())))
    return records

def main():
    toy=toy_checks();data=load_inner(0);records=[]
    for name,cfg in CONFIGS.items():
        seed_all(42);plain=make_adapter(data,'unmasked10' if cfg['reconstruction_weight'] else 'control','cuda:0')
        seed_all(42);a=make_adapter(data,name,'cuda:0');m=a.model
        for n,p in plain.model.state_dict().items():torch.testing.assert_close(p,m.state_dict()[n],atol=0,rtol=0)
        m.eval();plain.model.eval()
        with torch.no_grad():p0,p1=plain.predict('val'),a.predict('val')
        np.testing.assert_allclose(p0,p1,atol=1e-6,rtol=1e-6)
        # Nonzero weights make both conditioning and branch deletion observable.
        with torch.no_grad():
            m.context_projection.weight.normal_(std=.005)
            m.expression_context.modulator[-1].weight.normal_(std=.005)
        m.train();opt=make_optimizer(m,cfg);opt.zero_grad();torch.cuda.reset_peak_memory_stats();start=time.time()
        loss=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)
        if cfg['reconstruction_weight']:
            ids=m.reconstruction.masks(a.graph,.5,mask_generator('cuda:0',20261006));rec,_=m.reconstruction.loss(m.encoder,a.graph,ids,False);loss=loss+10*rec
        loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        assert all(p.grad is not None and p.grad.abs().max()>0 for p in m.expression_context.modulator.parameters())
        opt.step();torch.cuda.synchronize();elapsed=time.time()-start;peak=torch.cuda.max_memory_allocated()/2**30;m.eval()
        assert not m.lin1._forward_hooks and set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in n.lower() for n in m.state_dict())
        records.append(dict(config=name,initial_prediction_error=float(abs(p0-p1).max()),training_step_seconds=elapsed,peak_cuda_gib=peak,
            branch_deletion=real_branch_check(a,data),explicit_deletions=deletion_check(a,data,name,count=4)))
        print('PASS',name,elapsed,peak,flush=True);del a,plain;torch.cuda.empty_cache()
    # Every allocated parameter is equal across the matched condition arms.
    seed_all(42);a=make_adapter(data,'conditioned_rec10','cuda:0');seed_all(42);b=make_adapter(data,'mean_rec10','cuda:0')
    for n,p in a.model.state_dict().items():torch.testing.assert_close(p,b.model.state_dict()[n],atol=0,rtol=0)
    seed_all(42);b=make_adapter({**data,'val_y':1-data['val_y']},'conditioned_rec10','cuda:0')
    with torch.no_grad():
        a.model.context_projection.weight.normal_(std=.005);a.model.expression_context.modulator[-1].weight.normal_(std=.005)
        b.model.load_state_dict(a.model.state_dict())
    a.model.eval();b.model.eval()
    with torch.no_grad():pa,pb=a.predict('val'),b.predict('val')
    np.testing.assert_allclose(pa,pb,atol=2e-5,rtol=2e-5)
    result=dict(status='passed',toy=toy,records=records,matched_initial_parameters_equal=True,heldout_label_invariance=True,
        heldout_prediction_error=float(abs(pa-pb).max()),query_check_source_sha256=sha256(QUERY_CHECK_SOURCE),no_lncrna=True,
        outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'initial_checks.json').write_text(json.dumps(result,indent=2));print('ALL PASSED',flush=True)

if __name__=='__main__':main()
