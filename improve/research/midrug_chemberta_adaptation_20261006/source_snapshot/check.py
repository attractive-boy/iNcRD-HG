"""LoRA gradients, frozen-weight identity, feature contexts and physical deletion."""
import copy,json,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import HERE,DEGREE,read_module,load_inner,seed_all,sha256
from model import CONFIGS,LoRALinear,make_adapter,mask_generator,make_optimizers
QUERY_CHECK_SOURCE=DEGREE/'check.py'
_query=read_module('chemical_adaptation_query_checks',QUERY_CHECK_SOURCE);_query.make_adapter=make_adapter
deletion_check=_query.deletion_check

def algebra_check():
    seed_all(21);base=nn.Linear(7,5).double().requires_grad_(False);m=LoRALinear(base,3).double()
    with torch.no_grad():m.up.weight.normal_(std=.1)
    x=torch.randn(4,7,dtype=torch.double,requires_grad=True)
    z=m(x);ref=F.linear(x,base.weight+m.up.weight@m.down.weight,base.bias)
    torch.testing.assert_close(z,ref,atol=1e-12,rtol=1e-12)
    params=[x,m.down.weight,m.up.weight];ga=torch.autograd.grad(z.square().sum(),params);gb=torch.autograd.grad(ref.square().sum(),params)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-11)
    return dict(output_error=float(abs(z-ref).max()),gradient_error=max(float(abs(a-b).max()) for a,b in zip(ga,gb)))

@torch.no_grad()
def merged_block_check(adapt):
    if adapt.mode!='token':return None
    m=copy.deepcopy(adapt.block)
    for name in ['query','value']:
        old=getattr(m.attention.self,name);new=nn.Linear(768,768).to(old.base.weight)
        new.weight.copy_(old.base.weight+old.up.weight@old.down.weight);new.bias.copy_(old.base.bias)
        setattr(m.attention.self,name,new)
    m.eval();errors=[]
    for ix,h,mask in adapt.groups[:2]:
        extended=(1-mask[:,None,None,:].to(h.dtype))*torch.finfo(h.dtype).min
        z=adapt.block(h,attention_mask=extended)[0];ref=m(h,attention_mask=extended)[0]
        torch.testing.assert_close(z,ref,atol=2e-5,rtol=2e-5);errors.append(float(abs(z-ref).max()))
    return max(errors)

def frozen_parameters(a):return {n:p.detach().cpu().clone() for n,p in a.model.encoder.drug_adaptation.named_parameters() if not p.requires_grad}

def main():
    algebra=algebra_check();data=load_inner(0);records=[]
    for name,cfg in CONFIGS.items():
        seed_all(42);plain=make_adapter(data,'unmasked10' if cfg['reconstruction_weight'] else 'control','cuda:0')
        seed_all(42);a=make_adapter(data,name,'cuda:0');m=a.model;ad=m.encoder.drug_adaptation
        for n,t in plain.model.state_dict().items():torch.testing.assert_close(t,m.state_dict()[n],atol=0,rtol=0)
        m.eval();plain.model.eval()
        with torch.no_grad():p0,p1=plain.predict('val'),a.predict('val');x=ad(a.graph['drug'].x)
        np.testing.assert_allclose(p0,p1,atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(x,a.graph['drug'].x,atol=1e-7,rtol=1e-7)
        frozen=frozen_parameters(a)
        with torch.no_grad():
            for n,p in ad.named_parameters():
                if p.requires_grad and 'up' in n:p.normal_(std=.0001)
        merged=merged_block_check(ad)
        opts=make_optimizers(m,cfg);m.train();assert not ad.block.training
        for opt in opts:opt.zero_grad()
        torch.cuda.reset_peak_memory_stats();started=time.time()
        with m.encoder.feature_context():
            first=m.encoder._adapted_drug
            with m.encoder.feature_context():assert m.encoder._adapted_drug is first
            logits=a.logits('train');loss=F.binary_cross_entropy_with_logits(logits,a.y)
            if cfg['reconstruction_weight']:
                ids=m.reconstruction.masks(a.graph,.5,mask_generator('cuda:0',20261006));rec,_=m.reconstruction.loss(m.encoder,a.graph,ids,False);loss=loss+10*rec
            loss.backward()
        assert m.encoder._adapted_drug is None
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in ad.adaptation_parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        for opt in opts:opt.step()
        torch.cuda.synchronize();elapsed=time.time()-started;peak=torch.cuda.max_memory_allocated()/2**30
        for n,p in ad.named_parameters():
            if n in frozen:torch.testing.assert_close(p.cpu(),frozen[n],atol=0,rtol=0)
        m.eval();check=deletion_check(a,data,name,count=4)
        assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in n.lower() for n in m.state_dict())
        records.append(dict(config=name,initial_prediction_error=float(abs(p0-p1).max()),initial_feature_error=float(abs(x-a.graph['drug'].x).max()),
            trainable_adaptation_parameters=sum(p.numel() for p in ad.adaptation_parameters()),frozen_parameters_unchanged=True,
            merged_block_max_error=merged,training_step_seconds=elapsed,peak_cuda_gib=peak,explicit_deletions=check))
        print('PASS',name,elapsed,peak,flush=True);del a,plain;torch.cuda.empty_cache()
    seed_all(42);a=make_adapter(data,'token_rec10','cuda:0');seed_all(42);b=make_adapter({**data,'val_y':1-data['val_y']},'token_rec10','cuda:0')
    with torch.no_grad():
        for n,p in a.model.encoder.drug_adaptation.named_parameters():
            if p.requires_grad and 'up' in n:p.normal_(std=.0001)
        b.model.load_state_dict(a.model.state_dict())
    a.model.eval();b.model.eval()
    with torch.no_grad():pa,pb=a.predict('val'),b.predict('val')
    np.testing.assert_allclose(pa,pb,atol=2e-6,rtol=2e-6)
    assert all(r['trainable_adaptation_parameters']==24576 for r in records)
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',algebra=algebra,records=records,heldout_label_invariance=True,
        heldout_prediction_error=float(abs(pa-pb).max()),query_check_source_sha256=sha256(QUERY_CHECK_SOURCE),no_lncrna=True,
        outer_validation_evaluated=False,test_evaluated=False),indent=2));print('ALL PASSED',flush=True)

if __name__=='__main__':main()
