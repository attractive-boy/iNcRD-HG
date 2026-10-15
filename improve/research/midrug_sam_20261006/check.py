"""Verify the optimizer independently, including exact BN/parameter restoration."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import HERE,DEGREE,load_inner,seed_all,read_module,sha256
from model import CONFIGS,make_adapter,make_optimizer,generator,closure_for,perturbation_step
QUERY_CHECK_SOURCE=DEGREE/'check.py'
_query=read_module('original_m123_query_checks_for_sam',QUERY_CHECK_SOURCE)
# This helper calls its module's make_adapter; bind the unchanged pure M123 one.
_query.make_adapter=make_adapter
deletion_check=_query.deletion_check

def toy_checks():
    m=nn.Linear(2,1,bias=False).double();m.weight.data.copy_(torch.tensor([[.3,-.7]],dtype=torch.double))
    old=m.weight.detach().clone();target=torch.tensor([[.8,.2]],dtype=torch.double);rho=.05;lr=.03
    def closure():
        loss=.5*((m.weight-target)**2).sum();return loss,dict(loss=float(loss.detach()))
    opt=torch.optim.SGD(m.parameters(),lr=lr);g=old-target;delta=rho*g/g.norm();expected=old-lr*(g+delta)
    r=perturbation_step(m,opt,closure,rho,'sam')
    torch.testing.assert_close(m.weight,expected,atol=1e-14,rtol=1e-14)
    quadratic_error=float(abs(m.weight-expected).max())
    assert abs(r['alignment']-1)<1e-10 and abs(r['perturbation_norm']-rho)<1e-10
    # Failure in the second closure restores first-pass buffers and exact weights.
    m=nn.Sequential(nn.Linear(2,2),nn.BatchNorm1d(2),nn.Linear(2,1));m.train();opt=torch.optim.SGD(m.parameters(),lr=.1)
    x=torch.randn(8,2);y=torch.randn(8,1);calls=0;before={n:p.detach().clone() for n,p in m.named_parameters()};saved={}
    def failing():
        nonlocal calls
        calls+=1;loss=F.mse_loss(m(x),y)
        if calls==1:saved.update({n:b.detach().clone() for n,b in m.named_buffers()})
        if calls==2:raise RuntimeError('expected second-pass failure')
        return loss,dict(loss=float(loss.detach()))
    try:perturbation_step(m,opt,failing,.05,'sam')
    except RuntimeError as e:assert str(e)=='expected second-pass failure'
    else:raise AssertionError('Missing failure')
    for n,p in m.named_parameters():torch.testing.assert_close(p,before[n],atol=0,rtol=0)
    for n,b in m.named_buffers():torch.testing.assert_close(b,saved[n],atol=0,rtol=0)
    assert int(m[1].num_batches_tracked)==1
    return dict(quadratic_max_error=quadratic_error,second_pass_failure_restores_parameters_and_buffers=True)

def real_zero_check(data,name):
    # CPU reduces CUDA sparse/index accumulation noise; float32 backward accumulation
    # can still change at ~1e-7 when existing .grad buffers are zeroed and reused.
    seed_all(42);a=make_adapter(data,name,'cpu');seed_all(42);b=make_adapter(data,name,'cpu')
    a.model.train();b.model.train();o1,o2=make_optimizer(a.model),make_optimizer(b.model)
    idx=None;weight=0.
    if name.endswith('rec10'):idx=a.model.reconstruction.masks(a.graph,.5,generator('cpu',20261006));weight=.2
    c1,c2=closure_for(a,idx,weight),closure_for(b,idx,weight)
    o1.zero_grad();loss,_=c1();loss.backward();o1.step()
    perturbation_step(b.model,o2,c2,0.,'sam')
    errors={}
    for n,t in a.model.state_dict().items():
        v=b.model.state_dict()[n];torch.testing.assert_close(t,v,atol=2e-6,rtol=2e-5);errors[n]=float(abs(t-v).max())
        if 'running_' in n or 'num_batches_tracked' in n:torch.testing.assert_close(t,v,atol=0,rtol=0)
    for p1,p2 in zip(a.model.parameters(),b.model.parameters()):
        for n,t in o1.state[p1].items():torch.testing.assert_close(t,o2.state[p2][n],atol=2e-6,rtol=2e-5)
    return max(errors.values())

def independent_step(a,cfg,data,name):
    """Independently verify direction, perturbed gradient, exact restoration/update.

    Near-zero gradients can magnify float32 roundoff through RMSprop. Compare
    independently computed gradients before RMSprop; then check its update using
    the same verified gradient vector. Do not hide this as a looser weight tolerance.
    """
    m=a.model;m.train();b=make_adapter(data,name,a.y.device);b.model.load_state_dict(m.state_dict());b.model.train()
    opt=make_optimizer(m);ob=make_optimizer(b.model);base_state={n:t.clone() for n,t in m.state_dict().items()}
    idx=None;weight=0.
    if cfg['reconstruction_weight']:idx=m.reconstruction.masks(a.graph,.5,generator(a.y.device,5));weight=.2
    params=list(b.model.named_parameters());cb=closure_for(b,idx,weight)
    loss,_=cb();grads=torch.autograd.grad(loss,[p for _,p in params],allow_unused=True)
    eligible=[(n,p,g) for (n,p),g in zip(params,grads) if g is not None]
    rg=generator(a.y.device,12345)
    vs=[g if cfg['direction']=='sam' else torch.randn(p.shape,device=p.device,dtype=p.dtype,generator=rg) for n,p,g in eligible]
    denom=torch.cat([v.flatten() for v in vs]).norm()
    expected={n:v*(cfg['rho']/(denom+1e-12)) for (n,p,g),v in zip(eligible,vs)}
    captured={};errors={}
    def inspect(perturbed,originals,deltas):
        errors['direction_max_error']=0.
        for n,d in deltas.items():
            torch.testing.assert_close(d,expected[n],atol=2e-7,rtol=2e-5)
            torch.testing.assert_close(originals[n],base_state[n],atol=0,rtol=0)
            errors['direction_max_error']=max(errors['direction_max_error'],float(abs(d-expected[n]).max()))
        captured['buffers']={n:t.clone() for n,t in perturbed.named_buffers()}
        # Use exactly the actual perturbed weights to isolate gradient verification
        # from tiny discrepancies between equivalent global-norm reductions.
        b.model.load_state_dict(perturbed.state_dict());loss2,_=cb()
        g2=torch.autograd.grad(loss2,[p for _,p in params],allow_unused=True)
        captured['reference_gradients']={n:g for (n,p),g in zip(params,g2) if g is not None}
    real_step=opt.step
    def verified_step():
        errors['gradient_max_error']=0.
        for n,p in m.named_parameters():
            torch.testing.assert_close(p,base_state[n],atol=0,rtol=0)
            if p.grad is not None:
                ref=captured['reference_gradients'][n]
                torch.testing.assert_close(p.grad,ref,atol=2e-6,rtol=2e-5)
                errors['gradient_max_error']=max(errors['gradient_max_error'],float(abs(p.grad-ref).max()))
        for n,t in m.named_buffers():torch.testing.assert_close(t,captured['buffers'][n],atol=0,rtol=0)
        # Check RMSprop's update after exact restoration with the verified gradients.
        b.model.load_state_dict(m.state_dict())
        for (n,p),(nn,pp) in zip(m.named_parameters(),b.model.named_parameters()):
            assert n==nn;pp.grad=None if p.grad is None else p.grad.clone()
        ob.step();real_step()
        for n,t in m.state_dict().items():torch.testing.assert_close(t,b.model.state_dict()[n],atol=0,rtol=0)
        errors['restored_update_max_state_error']=0.
    opt.step=verified_step
    stats=perturbation_step(m,opt,closure_for(a,idx,weight),cfg['rho'],cfg['direction'],generator(a.y.device,12345),inspect)
    return dict(**errors,stats=stats,batchnorm_updates=int(m.eb.num_batches_tracked))


def main():
    seed_all(42);toy=toy_checks();data=load_inner(0)
    zero={n:real_zero_check(data,n) for n in ['sam','sam_rec10']};records=[]
    for n,cfg in CONFIGS.items():
        seed_all(42);a=make_adapter(data,n,'cuda:0');stats=independent_step(a,cfg,data,n)
        assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in k.lower() for k in a.model.state_dict())
        assert stats['batchnorm_updates']==1
        records.append(dict(config=n,**stats,explicit_deletions=deletion_check(a,data,n)));print('PASS',n,flush=True)
    seed_all(42);a=make_adapter(data,'sam_rec10','cuda:0');seed_all(42);b=make_adapter({**data,'val_y':1-data['val_y']},'sam_rec10','cuda:0')
    for n,t in a.model.state_dict().items():torch.testing.assert_close(t,b.model.state_dict()[n],atol=0,rtol=0)
    for n in a.graph.node_types:torch.testing.assert_close(a.graph[n].x,b.graph[n].x,atol=0,rtol=0)
    for e in a.graph.edge_types:torch.testing.assert_close(a.graph[e].edge_index,b.graph[e].edge_index,atol=0,rtol=0)
    for part in a.qis:
        torch.testing.assert_close(a.qis[part],b.qis[part],atol=0,rtol=0)
        torch.testing.assert_close(a.ease[part],b.ease[part],atol=0,rtol=0)
    a.model.eval();b.model.eval()
    with torch.no_grad():pa,pb=a.predict('val'),b.predict('val')
    np.testing.assert_allclose(pa,pb,atol=1e-6,rtol=1e-6)
    result=dict(status='passed',toy=toy,zero_radius_state_errors=zero,zero_radius_device='cpu',records=records,heldout_label_invariance=True,
        heldout_prediction_max_error=float(abs(pa-pb).max()),no_lncrna=True,query_check_source_sha256=sha256(QUERY_CHECK_SOURCE))
    (HERE/'initial_checks.json').write_text(json.dumps(result,indent=2));print('ALL PASSED',flush=True)

if __name__=='__main__':main()
