"""Independent interpolation derivatives, partner boundaries and model checks."""
import ast,json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import HERE,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,make_optimizer,mask_generator,mixing_plan,interpolate,capture_features,decode,association_loss
QUERY_SOURCE=HERE.parent/'midrug_degree_encoder_20261006/check.py'
tree=ast.parse(QUERY_SOURCE.read_text());body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='deletion_check']
ns=dict(np=np,torch=torch,make_adapter=make_adapter);exec(compile(ast.Module(body=body,type_ignores=[]),str(QUERY_SOURCE),'exec'),ns);deletion_check=ns['deletion_check']

def sampler_check(pairs,fold):
    result=[]
    for epoch in [1,31,60,400,800]:
        first=None
        for mode in ['drug','global']:
            pi,lam,active=mixing_plan(pairs,mode,fold,epoch);repeat=mixing_plan(pairs,mode,fold,epoch)
            np.testing.assert_array_equal(np.sort(pi),np.arange(len(pi)));np.testing.assert_array_equal(pi,repeat[0]);assert lam==repeat[1]
            assert np.all(pi[active]!=active)
            inactive=np.setdiff1d(np.arange(len(pi)),active);np.testing.assert_array_equal(pi[inactive],inactive)
            if mode=='drug':np.testing.assert_array_equal(pairs[:,1],pairs[pi,1])
            if first is None:first=(lam,active)
            else:assert lam==first[0];np.testing.assert_array_equal(active,first[1])
            result.append(dict(epoch=epoch,mode=mode,lambda_value=lam,eligible=len(active),singletons=len(inactive)))
    return result

def mathematical_check():
    torch.manual_seed(50);h=torch.randn(6,7,dtype=torch.double,requires_grad=True);y=torch.tensor([0.,1.,0.,1.,1.,0.],dtype=torch.double)
    pi=np.array([1,0,3,4,2,5]);lam=.37
    net=nn.Sequential(nn.Linear(7,5),nn.Tanh(),nn.Linear(5,1)).double()
    hh,yy=interpolate(h,y,pi,lam);loss=F.binary_cross_entropy_with_logits(net(hh).ravel(),yy)
    reference=torch.stack([F.binary_cross_entropy_with_logits(net(lam*h[i]+(1-lam)*h[int(pi[i])]).ravel(),(lam*y[i]+(1-lam)*y[int(pi[i])]).view(1)) for i in range(len(h))]).mean()
    parameters=[h,*net.parameters()];actual=torch.autograd.grad(loss,parameters,retain_graph=True);expected=torch.autograd.grad(reference,parameters,retain_graph=True)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=1e-13,rtol=1e-13)
    # Independent logaddexp loss evaluation, not the vectorized torch mixing path.
    logits=net(hh).detach().numpy().ravel();labels=np.array([lam*float(y[i])+(1-lam)*float(y[pi[i]]) for i in range(len(y))])
    oracle=np.mean(np.logaddexp(0.,logits)-labels*logits);assert abs(float(loss)-oracle)<1e-13
    assert abs(float(yy.mean()-y.mean()))<1e-15
    # An anchor loss must send gradient to its distinct partner as well.
    anchor=F.binary_cross_entropy_with_logits(net(hh[:1]).ravel(),yy[:1])
    g=torch.autograd.grad(anchor,h)[0];assert g[0].abs().max()>0 and g[1].abs().max()>0 and g[2:].abs().max()==0
    for weight in [0.,1.]:
        mixed,target=interpolate(h,y,pi,weight);ix=np.arange(len(h)) if weight else pi
        torch.testing.assert_close(mixed,h[ix],atol=0,rtol=0);torch.testing.assert_close(target,y[ix],atol=0,rtol=0)
    toy=np.array([[0,1],[1,1],[2,2],[3,3],[4,3],[5,3]])
    sampler=sampler_check(toy,0);assert sampler[0]['singletons']==1
    return dict(loss_error=abs(float(loss)-oracle),gradient_error=max(float(abs(a-b).max()) for a,b in zip(actual,expected)),both_source_gradients=True,toy_sampler=sampler)

def main():
    math=mathematical_check();data=load_inner(0);sampling=[sampler_check(load_inner(f)['train_pairs'],f) for f in range(3)];records=[]
    for name,cfg in CONFIGS.items():
        seed_all(42);a=make_adapter(data,name,'cuda:0');seed_all(42);b=make_adapter(data,'unmasked10','cuda:0')
        for k,v in a.model.state_dict().items():torch.testing.assert_close(v,b.model.state_dict()[k],atol=0,rtol=0)
        a.model.train();b.model.train();opt=make_optimizer(a.model);opt.zero_grad(set_to_none=True)
        loss,stats,extra=association_loss(a,cfg,1,0)
        plain=F.binary_cross_entropy_with_logits(b.logits('train'),b.y)
        loss.backward();plain.backward()
        errors=[]
        for (k,p),(l,q) in zip(a.model.named_parameters(),b.model.named_parameters()):
            assert k==l and ((p.grad is None)==(q.grad is None))
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-5);errors.append(float(abs(p.grad-q.grad).max()))
        assert int(a.model.eb.num_batches_tracked)==1
        del b;opt.zero_grad(set_to_none=True);loss,stats,extra=association_loss(a,cfg,60,0)
        with torch.no_grad():direct=decode(a.model,extra['features'])
        torch.testing.assert_close(direct,extra['logits'],atol=1e-6,rtol=1e-6)
        idx=a.model.reconstruction.masks(a.graph,.5,mask_generator('cuda:0',20261006));rec,_=a.model.reconstruction.loss(a.model.encoder,a.graph,idx,False)
        (loss+10*rec).backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.model.parameters())
        assert any(p.grad is not None and p.grad.abs().max()>0 for p in a.model.encoder.parameters());opt.step();a.model.eval()
        assert int(a.model.eb.num_batches_tracked)==2 and not a.model.lin1._forward_pre_hooks
        deleted=deletion_check(a,data,name,count=4)
        b=make_adapter({**data,'val_y':1-data['val_y']},name,'cuda:0');b.model.load_state_dict(a.model.state_dict());b.model.eval()
        for mode in ['drug','global']:
            pa=mixing_plan(a.mix_pairs,mode,0,60);pb=mixing_plan(b.mix_pairs,mode,0,60);np.testing.assert_array_equal(pa[0],pb[0]);assert pa[1]==pb[1]
        with torch.no_grad():pa,pb=a.predict('val'),b.predict('val')
        np.testing.assert_allclose(pa,pb,atol=2e-5,rtol=2e-5)
        records.append(dict(config=name,zero_weight_gradient_error=max(errors),stats=stats,explicit_deletions=deleted,heldout_flip_error=float(abs(pa-pb).max())))
        print('PASS',name,flush=True);del a,b,opt;torch.cuda.empty_cache()
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',mathematics=math,real_sampler_checks=sampling,records=records,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
