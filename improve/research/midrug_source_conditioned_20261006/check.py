"""Independent dense propagation after physical deletion, including gradients."""
import json,time
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,DEGREE,read_module,load_inner,seed_all,sha256
from model import CONFIGS,SourcePropagation,neighbor_list,make_adapter,mask_generator,make_optimizer
QUERY_CHECK_SOURCE=DEGREE/'check.py'
_query=read_module('query_checks_for_source_conditioning',QUERY_CHECK_SOURCE);_query.make_adapter=make_adapter
deletion_check=_query.deletion_check

def independent_deleted(m,A,Sr,Sd,r,d):
    """One source context; ordinary dense matmuls, no stored states or correction."""
    A=A.clone();A[r,d]=0;nr,nd=A.shape
    hm=A.new_zeros(nr,m.hidden);marker=A.new_zeros(nd)
    if m.conditioned:marker[d]=1.
    else:marker.fill_(nd**-.5)
    hd=marker[:,None]*m.q;outs=[]
    for layer in range(3):
        pm=A@m.dm[layer](hd)+Sr@m.mm[layer](hm)+m.sm[layer](hm)
        pd=A.T@m.md[layer](hm)+Sd@m.dd[layer](hd)+m.sd[layer](hd)+marker[:,None]*m.boundary[layer]
        hm=F.leaky_relu(F.layer_norm(pm,(m.hidden,),m.nm[layer].weight,m.nm[layer].bias,m.nm[layer].eps),.1)
        hd=F.leaky_relu(F.layer_norm(pd,(m.hidden,),m.nd[layer].weight,m.nd[layer].bias,m.nd[layer].eps),.1)
        outs.append(hm[r])
    return torch.cat(outs[1:])

def toy_checks():
    # Weighted associations, directed same-type edges, an isolated/degree-one miRNA.
    A=torch.tensor([[1.,1.,0.,0.],[0.,1.,1.,0.],[2.,0.,1.,0.],[0.,0.,0.,1.]],dtype=torch.double)
    Sr=torch.tensor([[0.,2.,0.,0.],[1.,0.,1.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]],dtype=torch.double)
    Sd=torch.tensor([[0.,1.,0.,0.],[0.,0.,1.,0.],[1.,0.,0.,0.],[0.,0.,0.,0.]],dtype=torch.double)
    cases=[(0,0),(2,0),(3,3),(0,2),(3,0)];records=[]
    for condition in [True,False]:
        seed_all(19);m=SourcePropagation(5,condition).double();pars=list(m.parameters())
        for r,d in cases:
            out=m(A,Sr,Sd,neighbor_list(Sr),torch.tensor([r]),torch.tensor([d]))[0]
            reference=independent_deleted(m,A,Sr,Sd,r,d)
            torch.testing.assert_close(out,reference,atol=1e-10,rtol=1e-10)
            w=torch.linspace(-.7,.9,len(out),dtype=out.dtype)
            ga=torch.autograd.grad((out*w).sum(),pars,allow_unused=True);gb=torch.autograd.grad((reference*w).sum(),pars,allow_unused=True)
            errors=[]
            for g,h in zip(ga,gb):
                if g is None or h is None:
                    assert (g is None or torch.count_nonzero(g)==0) and (h is None or torch.count_nonzero(h)==0)
                else:torch.testing.assert_close(g,h,atol=2e-9,rtol=2e-9);errors.append(float(abs(g-h).max()))
            records.append(dict(conditioned=condition,pair=[r,d],removed_weight=float(A[r,d]),output_error=float(abs(out-reference).max()),gradient_error=max(errors)))
    return records

@torch.no_grad()
def real_branch_check(a,data,count=4):
    # The backbone's query invariant check is independent and complementary.
    ix=np.r_[np.flatnonzero(data['train_y']==1)[:count],np.flatnonzero(data['train_y']==0)[:2]]
    m=a.model;records=[]
    for i in ix:
        r,d=map(int,data['train_pairs'][i]);t=m.source_path(a.ps.A,m.source_sr,m.source_sd,m.source_nbr,
            torch.tensor([r],device=a.y.device),torch.tensor([d],device=a.y.device))[0]
        ref=independent_deleted(m.source_path,a.ps.A,m.source_sr,m.source_sd,r,d)
        torch.testing.assert_close(t,ref,atol=2e-4,rtol=5e-5)
        records.append(dict(pair=[r,d],label=int(data['train_y'][i]),max_error=float(abs(t-ref).max())))
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
        # Nonzero new weights make the full-model deletion checks sensitive to branch errors.
        with torch.no_grad():m.source_projection.weight.normal_(std=.005)
        m.train();opt=make_optimizer(m,cfg);opt.zero_grad();started=time.time();torch.cuda.reset_peak_memory_stats()
        logits=a.logits('train');bce=F.binary_cross_entropy_with_logits(logits,a.y);loss=bce
        if cfg['reconstruction_weight']:
            ids=m.reconstruction.masks(a.graph,.5,mask_generator('cuda:0',20261006));rec,_=m.reconstruction.loss(m.encoder,a.graph,ids,False);loss=loss+10*rec
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        assert m.source_path.q.grad.abs().max()>0 and m.source_projection.weight.grad.abs().max()>0
        opt.step();torch.cuda.synchronize();elapsed=time.time()-started;peak=torch.cuda.max_memory_allocated()/2**30;m.eval()
        assert not m.lin1._forward_hooks
        assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in n.lower() for n in m.state_dict())
        records.append(dict(config=name,initial_prediction_error=float(abs(p0-p1).max()),training_step_seconds=elapsed,peak_cuda_gib=peak,
            branch_deletion=real_branch_check(a,data),explicit_deletions=deletion_check(a,data,name,count=4)))
        print('PASS',name,elapsed,peak,flush=True);del a,plain;torch.cuda.empty_cache()
    seed_all(42);a=make_adapter(data,'source_rec10','cuda:0');seed_all(42);b=make_adapter({**data,'val_y':1-data['val_y']},'source_rec10','cuda:0')
    with torch.no_grad():a.model.source_projection.weight.normal_(std=.005);b.model.load_state_dict(a.model.state_dict())
    for n,t in a.model.state_dict().items():torch.testing.assert_close(t,b.model.state_dict()[n],atol=0,rtol=0)
    for n in ['source_sr','source_sd','source_nbr']:torch.testing.assert_close(getattr(a.model,n),getattr(b.model,n),atol=0,rtol=0)
    a.model.eval();b.model.eval()
    with torch.no_grad():pa,pb=a.predict('val'),b.predict('val')
    np.testing.assert_allclose(pa,pb,atol=2e-5,rtol=2e-5)
    result=dict(status='passed',toy=toy,records=records,heldout_label_invariance=True,heldout_prediction_max_error=float(abs(pa-pb).max()),
        query_check_source_sha256=sha256(QUERY_CHECK_SOURCE),no_lncrna=True)
    (HERE/'initial_checks.json').write_text(json.dumps(result,indent=2));print('ALL PASSED',flush=True)

if __name__=='__main__':main()
