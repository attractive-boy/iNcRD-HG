"""Independent value formula, own-label exclusion and full query deletions."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,read_module,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,make_optimizer,mask_generator
QUERY_CHECK_SOURCE=HERE.parent/'midrug_degree_encoder_20261006/check.py'
query=read_module('label_retrieval_deletions',QUERY_CHECK_SOURCE);query.make_adapter=make_adapter
deletion_check=query.deletion_check

@torch.no_grad()
def batch_check(a):
    batched=a.logits('val')[:3]
    alone=torch.cat([a.model(a.pairs['val'][:,i:i+1],a.ps,a.qis['val'][i:i+1],a.ease['val'][i:i+1],a.dense) for i in range(3)])
    torch.testing.assert_close(batched,alone,atol=2e-4,rtol=5e-5)
    return float(abs(batched-alone).max())

@torch.no_grad()
def memory_check(a):
    m=a.model.case_memory;pairs=a.pairs['train']
    selected=torch.cat([(a.y==label).nonzero().flatten()[:2] for label in [0,1]])
    q=pairs[:,selected];out=m(q).clone();records=[]
    for j,ix in enumerate(selected):
        old=m.support_y[ix].clone();m.support_y[ix]=1-old
        changed=m(q[:,j:j+1]);torch.testing.assert_close(out[j:j+1],changed,atol=2e-5,rtol=2e-5)
        m.support_y[ix]=old
        keep=torch.arange(len(m.support_y),device=q.device)!=ix
        saved=[m.support_pairs,m.support_y,m.support_ids]
        m.support_pairs,m.support_y,m.support_ids=[v[keep] for v in saved]
        deleted=m(q[:,j:j+1]);torch.testing.assert_close(out[j:j+1],deleted,atol=2e-5,rtol=2e-5)
        m.support_pairs,m.support_y,m.support_ids=saved
        records.append(dict(index=int(ix),label=int(old),flip_error=float(abs(out[j]-changed).max()),deletion_error=float(abs(out[j]-deleted).max())))
    h,pooled,ix,weights=m.context(q)
    if ix is not None:
        nodes=m.nodes();keys={n:F.linear(x,m.K.weight,None) for n,x in nodes.items()}
        qk=(keys['miRNA'][q[0]]+keys['drug'][q[1]]+m.K.bias).cpu().numpy().astype(np.float64)
        sp=m.support_pairs;sk=(keys['miRNA'][sp[:,0]]+keys['drug'][sp[:,1]]+m.K.bias).cpu().numpy().astype(np.float64)
        distance=np.sum((qk[:,None]-sk[None,:])**2,axis=2);distance[np.arange(len(selected)),selected.cpu().numpy()]=np.inf
        expected_ix=np.argsort(distance,axis=1,kind='stable')[:,:32]
        np.testing.assert_array_equal(ix.cpu(),expected_ix)
        # Explicit float32 subtraction matches the model's selected distance computation.
        delta=(qk.astype(np.float32)[:,None]-sk.astype(np.float32)[expected_ix]).astype(np.float64)
        sim=-np.sum(delta**2,axis=2);p=np.exp(sim-sim.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        value=np.maximum(0,delta@m.T[0].weight.cpu().numpy().T+m.T[0].bias.cpu().numpy())@m.T[2].weight.cpu().numpy().T
        if m.config['labels']:value+=m.label.weight.cpu().numpy()[m.support_y.cpu().numpy()[expected_ix]]
        expected=np.sum(p[...,None]*value,axis=1)
        np.testing.assert_allclose(pooled.cpu(),expected,atol=2e-5,rtol=2e-5)
        error=float(abs(pooled.cpu().numpy()-expected).max())
    else:error=0.;assert (pooled==0).all()
    return dict(own_label_records=records,numpy_value_max_error=error)

def main():
    records=[]
    for fold in range(3):
        data=load_inner(fold);seed_all(42);baseline=make_adapter(data,'unmasked10','cuda:0');baseline.model.eval()
        initial={k:v.clone() for k,v in baseline.model.state_dict().items()}
        with torch.no_grad():base=baseline.predict('val')
        for name in CONFIGS:
            seed_all(42);a=make_adapter(data,name,'cuda:0');m=a.model;m.eval()
            assert not any('support_' in k or 'x_miRNA' in k or 'x_drug' in k for k in m.state_dict())
            for k,v in initial.items():torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)
            with torch.no_grad():p=a.predict('val')
            np.testing.assert_allclose(p,base,atol=2e-5,rtol=2e-5)
            opt=make_optimizer(m);generator=mask_generator('cuda:0',20261006+fold)
            losses=[]
            for ep in [1,2]:
                m.train();opt.zero_grad(set_to_none=True);bce=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)
                rec=m.reconstruction;indices=rec.masks(a.graph,.5,generator);rl,_=rec.loss(m.encoder,a.graph,indices,False)
                loss=bce+(10*ep/50)*rl;loss.backward()
                assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
                opt.step();losses.append(float(loss))
            if name=='label':assert m.case_memory.label.weight.grad is not None and m.case_memory.label.weight.grad.abs().max()>0
            else:assert m.case_memory.label.weight.grad is None
            if name=='direct':assert all(p.grad is None for sub in [m.case_memory.T,m.case_memory.K] for p in sub.parameters())
            m.eval();records.append(dict(config=name,fold=fold,initial_error=float(abs(p-base).max()),losses=losses,
                memory_check=memory_check(a),batch_max_logit_error=batch_check(a),explicit_deletions=deletion_check(a,data,name,count=4)))
            print('PASS',name,fold,flush=True);del a,m,opt;torch.cuda.empty_cache()
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',records=records,
        source_hashes={str(p):sha256(p) for p in [HERE/'check.py',HERE/'common.py',HERE/'model.py',QUERY_CHECK_SOURCE]},
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
