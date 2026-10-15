"""Explicit graph deletion, including changes to degree denominators."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,load_inner,seed_all
from model import CONFIGS,make_adapter,mask_generator


@torch.no_grad()
def deletion_check(a,data,name,count=6):
    a.model.eval()
    pos=np.flatnonzero(data['train_y']==1)
    pairs=data['train_pairs'][pos]
    degree=np.bincount(pairs[:,0],minlength=a.graph['miRNA'].num_nodes)
    # Include the least-connected positive endpoint, often degree one -> zero.
    order=np.argsort(degree[pairs[:,0]],kind='stable')
    selected=np.unique(np.r_[pos[order[:2]],pos[np.linspace(0,len(pos)-1,count-2,dtype=int)]])
    neg=np.flatnonzero(data['train_y']==0)[:2]
    records=[]
    for ix in np.r_[selected,neg]:
        keep=np.arange(len(data['train_y']))!=ix
        other={**data,'train_pairs':data['train_pairs'][keep],'train_y':data['train_y'][keep]}
        b=make_adapter(other,name,a.y.device);b.model.load_state_dict(a.model.state_dict());b.model.eval()
        r,d=a.pairs['train'][:,ix:ix+1]
        z1=a.model.encoder.forward_dense(*a.dense,r,d,a.ps.qvals(r,d))
        z2=b.model.encoder.forward_dense(*b.dense,r,d,b.ps.qvals(r,d))
        errors={}
        for label,j in [('M1_miRNA',2),('M1_drug',3)]:
            torch.testing.assert_close(z1[j],z2[j],atol=2e-4,rtol=5e-5)
            errors[label]=float(abs(z1[j]-z2[j]).max())
        pool1=a.ps.qncn(r,d,{'miRNA':z1[0],'drug':z1[1]},z1[4])
        pool2=b.ps.qncn(r,d,{'miRNA':z2[0],'drug':z2[1]},None)
        for j,(v,w) in enumerate(zip(pool1,pool2)):
            torch.testing.assert_close(v,w,atol=2e-4,rtol=5e-5);errors[f'M2_{j}']=float(abs(v-w).max())
        qi=a.qis['train'][ix:ix+1];qi2=(b.ps.qis(r,d)-a.mu)/a.sd
        ease=a.ease['train'][ix:ix+1];ease2=b.ps.ease(r,d)
        torch.testing.assert_close(qi,qi2,atol=2e-4,rtol=5e-5)
        torch.testing.assert_close(ease,ease2,atol=2e-4,rtol=5e-5)
        out1=a.model(torch.stack((r,d)),a.ps,qi,ease,a.dense)
        out2=b.model(torch.stack((r,d)),b.ps,qi2,ease2,b.dense)
        torch.testing.assert_close(out1,out2,atol=2e-4,rtol=5e-5)
        errors['full_logit']=float(abs(out1-out2).max())
        records.append(dict(pair=data['train_pairs'][ix].tolist(),label=int(data['train_y'][ix]),
                            miRNA_positive_degree=int(degree[int(r)]),errors=errors))
    return records


def main():
    data=load_inner(0);seed_all(42);base=make_adapter(data,'control','cuda:0');base.model.eval()
    seed_all(42);identity=make_adapter(data,'identity','cuda:0');identity.model.eval()
    for n,p in base.model.state_dict().items():
        torch.testing.assert_close(p,identity.model.state_dict()[n],atol=0,rtol=0)
    with torch.no_grad():
        p0,p1=base.logits('train'),identity.logits('train')
    torch.testing.assert_close(p0,p1,atol=2e-4,rtol=5e-5)
    records=[]
    for name,cfg in CONFIGS.items():
        seed_all(42);a=make_adapter(data,name,'cuda:0');a.model.train()
        loss=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)
        if cfg['reconstruction_weight']:
            rec=a.model.reconstruction;idx=rec.masks(a.graph,.5,mask_generator('cuda:0',0))
            aux,_=rec.loss(a.model.encoder,a.graph,idx,False);loss=loss+cfg['reconstruction_weight']*aux
        opt=torch.optim.RMSprop(a.model.parameters(),lr=1e-4);opt.zero_grad();loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.model.parameters())
        assert any(p.grad is not None and p.grad.abs().max()>0 for p in a.model.encoder.parameters())
        opt.step()
        records.append(dict(config=name,finite_gradients=True,explicit_deletions=deletion_check(a,data,name)))
        print('PASS',name,flush=True)
    # Heldout labels do not determine any normalization term or raw pair channel.
    seed_all(42);a=make_adapter(data,'all05','cuda:0');a.model.eval()
    seed_all(42);b=make_adapter({**data,'val_y':1-data['val_y']},'all05','cuda:0');b.model.eval()
    for n,p in a.model.state_dict().items():
        torch.testing.assert_close(p,b.model.state_dict()[n],atol=0,rtol=0)
    for edge,scale in a.model.encoder.scales.items():
        torch.testing.assert_close(scale,b.model.encoder.scales[edge],atol=0,rtol=0)
    for part in a.qis:
        torch.testing.assert_close(a.qis[part],b.qis[part],atol=0,rtol=0)
        torch.testing.assert_close(a.ease[part],b.ease[part],atol=0,rtol=0)
    with torch.no_grad():
        pa,pb=a.predict('val'),b.predict('val')
        np.testing.assert_allclose(pa,pb,atol=1e-6,rtol=1e-6)
    result=dict(status='passed',identity_max_logit_error=float(abs(p0-p1).max()),
        zero_exponent_same_parameters=True,heldout_label_invariance=True,
        heldout_flip_max_prediction_error=float(abs(pa-pb).max()),records=records,
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'initial_checks.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
