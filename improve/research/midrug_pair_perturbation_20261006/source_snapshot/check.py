"""Independent ellipsoid oracle, frozen-direction derivatives and query deletion."""
import json,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import HERE,DEGREE,read_module,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,make_optimizer,mask_generator,capture_features,decode,feature_delta,association_loss
QUERY_CHECK_SOURCE=DEGREE/'check.py'
_query=read_module('original_query_check_pair_perturbation',QUERY_CHECK_SOURCE);_query.make_adapter=make_adapter
deletion_check=_query.deletion_check

def oracle_checks():
    torch.manual_seed(4);x=torch.randn(9,7,dtype=torch.double,requires_grad=True)
    w=torch.linspace(-.7,.5,7,dtype=torch.double);labels=(torch.arange(9)%2).double()
    logits=x@w+.2;loss=F.binary_cross_entropy_with_logits(logits,labels)
    gradient=torch.autograd.grad(loss,x,retain_graph=True)[0]
    oracle=(logits.detach().sigmoid()-labels)[:,None]*w/len(x)
    torch.testing.assert_close(gradient,oracle,atol=1e-14,rtol=1e-14)
    delta,info=feature_delta(x,gradient,.05,'adv');s=x.detach().numpy().std(axis=0).clip(.001)
    expected=[]
    for g in oracle.numpy():
        z=s*g;expected.append(.05*math.sqrt(7)*s*z/np.linalg.norm(z))
    np.testing.assert_allclose(delta.numpy(),np.array(expected),atol=1e-14,rtol=1e-14)
    assert not delta.requires_grad and not info['scale'].requires_grad
    # Cauchy-Schwarz maximum of each independent ellipsoid linear program.
    actual=(gradient.detach()*delta).sum(1)
    bound=.05*math.sqrt(7)*(gradient.detach()*torch.tensor(s)).norm(dim=1)
    torch.testing.assert_close(actual,bound,atol=1e-14,rtol=1e-14)
    random,ri=feature_delta(x,gradient,.05,'random',mask_generator('cpu',91))
    repeat,_=feature_delta(x,gradient,.05,'random',mask_generator('cpu',91));torch.testing.assert_close(random,repeat,atol=0,rtol=0)
    assert torch.all((gradient.detach()*random).sum(1)<=bound+1e-14)
    # Positive column rescaling/translation (away from floor) preserves normalized perturbation.
    c=torch.linspace(.5,2.,7,dtype=torch.double);shift=torch.arange(7,dtype=torch.double)
    rescaled,_=feature_delta(x*c+shift,gradient/c,.05,'adv')
    torch.testing.assert_close(rescaled,delta*c,atol=1e-14,rtol=1e-14)
    for mode in ['adv','random']:
        z,zi=feature_delta(x,torch.zeros_like(x),.05,mode,mask_generator('cpu',9))
        assert torch.count_nonzero(z)==0 and not zi['active'].any()
    # Nonlinear finite differences with the perturbation fixed (first-order training).
    net=nn.Sequential(nn.Linear(7,11),nn.Tanh(),nn.Linear(11,1)).double()
    h=x.detach().clone().requires_grad_();fn=lambda value:.5*(F.binary_cross_entropy_with_logits(net(value).view(-1),labels)+F.binary_cross_entropy_with_logits(net(value+delta).view(-1),labels))
    got=torch.autograd.grad(fn(h),h)[0];errors=[]
    for row,col in [(0,0),(2,3),(6,1),(8,6)]:
        plus=h.detach().clone();minus=h.detach().clone();plus[row,col]+=1e-6;minus[row,col]-=1e-6
        finite=(fn(plus)-fn(minus))/(2e-6);errors.append(float(abs(finite-got[row,col])))
    assert max(errors)<1e-9
    return dict(ellipsoid_max_error=float(abs(delta-torch.tensor(np.array(expected))).max()),max_finite_difference_error=max(errors),rescaling_equivariant=True,zero_gradient_policy_checked=True)

def main():
    seed_all(42);oracle=oracle_checks();data=load_inner(0);records=[]
    # Zero perturbation and zero mixture both reduce to ordinary BCE, including gradients.
    zero=[]
    for epoch,rms in [(60,0.),(1,.05)]:
        seed_all(42);a=make_adapter(data,'adv','cuda:0');a.model.train();a.model.zero_grad()
        loss,stats,extra=association_loss(a,CONFIGS['adv'],epoch,mask_generator('cuda:0',3),rms_override=rms)
        pars=list(a.model.parameters());expected=torch.autograd.grad(extra['clean'],pars,retain_graph=True,allow_unused=True)
        got=torch.autograd.grad(loss,pars,allow_unused=True)
        errs=[]
        for g,v in zip(got,expected):
            assert (g is None)==(v is None)
            if g is not None:torch.testing.assert_close(g,v,atol=2e-6,rtol=2e-5);errs.append(float(abs(g-v).max()))
        assert int(a.model.eb.num_batches_tracked)==1
        zero.append(dict(epoch=epoch,rms=rms,max_gradient_error=max(errs),loss_error=float(abs(loss-extra['clean']))))
    for name,cfg in CONFIGS.items():
        seed_all(42);a=make_adapter(data,name,'cuda:0');m=a.model;m.train();opt=make_optimizer(m);opt.zero_grad()
        before={n:p.detach().clone() for n,p in m.named_parameters()}
        loss,stats,extra=association_loss(a,cfg,60,mask_generator('cuda:0',123))
        with torch.no_grad():direct=decode(m,extra['features'])
        torch.testing.assert_close(direct,extra['logits'],atol=1e-6,rtol=1e-6)
        for n,p in m.named_parameters():torch.testing.assert_close(p,before[n],atol=0,rtol=0);assert p.grad is None
        if cfg['reconstruction_weight']:
            indices=m.reconstruction.masks(a.graph,.5,mask_generator('cuda:0',20261006))
            rec,_=m.reconstruction.loss(m.encoder,a.graph,indices,False);loss=loss+10.*rec
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        assert any(p.grad is not None and p.grad.abs().max()>0 for p in m.encoder.parameters())
        assert int(m.eb.num_batches_tracked)==1 and not m.lin1._forward_pre_hooks
        assert stats['normalized_rms_max_error']<1e-7
        if cfg['direction']=='adv':assert abs(stats['alignment_mean']-1)<1e-5
        opt.step();m.eval();checks=deletion_check(a,data,name)
        assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in n.lower() for n in m.state_dict())
        records.append(dict(config=name,stats=stats,decode_max_error=float(abs(direct-extra['logits']).max()),explicit_deletions=checks))
        print('PASS',name,flush=True)
    seed_all(42);a=make_adapter(data,'adv_rec10','cuda:0');seed_all(42);b=make_adapter({**data,'val_y':1-data['val_y']},'adv_rec10','cuda:0')
    for n,t in a.model.state_dict().items():torch.testing.assert_close(t,b.model.state_dict()[n],atol=0,rtol=0)
    for e in a.graph.edge_types:torch.testing.assert_close(a.graph[e].edge_index,b.graph[e].edge_index,atol=0,rtol=0)
    for part in a.qis:
        torch.testing.assert_close(a.qis[part],b.qis[part],atol=0,rtol=0);torch.testing.assert_close(a.ease[part],b.ease[part],atol=0,rtol=0)
    a.model.eval();b.model.eval()
    with torch.no_grad():pa,pb=a.predict('val'),b.predict('val')
    np.testing.assert_allclose(pa,pb,atol=1e-6,rtol=1e-6)
    result=dict(status='passed',oracle=oracle,zero_effect_checks=zero,records=records,heldout_label_invariance=True,heldout_prediction_max_error=float(abs(pa-pb).max()),
        query_check_source_sha256=sha256(QUERY_CHECK_SOURCE),no_lncrna=True)
    (HERE/'initial_checks.json').write_text(json.dumps(result,indent=2));print('ALL PASSED',flush=True)

if __name__=='__main__':main()
