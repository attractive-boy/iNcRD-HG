"""Member-matrix/gradient reference, exact fusion identity and memberwise deletion."""
import json,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import HERE,DEGREE,read_module,load_inner,seed_all,sha256
from model import (CONFIGS,EnsembleLinear,EnsembleOutput,make_adapter,member_logits,
                   classification_loss,mixture_logit,make_optimizer,mask_generator)

QUERY_CHECK_SOURCE=DEGREE/'check.py'
query=read_module('ensemble_fusion_original_deletion',QUERY_CHECK_SOURCE)
query.make_adapter=make_adapter


class ToyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.first=EnsembleLinear(nn.Linear(5,7).double(),4,True)
        self.second=EnsembleLinear(nn.Linear(7,3).double(),4)
        self.output=EnsembleOutput(nn.Linear(3,1).double(),4)

    def forward(self,x):
        z=F.leaky_relu(self.first(x[:,None,:].expand(-1,4,-1)),.1)
        z=F.leaky_relu(self.second(z),.1)
        return self.output(z)


def oracle_head(head,x):
    outputs=[]
    for i in range(4):
        h=x
        for layer in [head.first,head.second]:
            matrix=torch.diag(layer.s[i])@layer.weight@torch.diag(layer.r[i])
            h=h@matrix.T+layer.bias[i]
            h=torch.where(h>=0,h,.1*h)
        outputs.append(h@head.output.weight[i]+head.output.bias[i])
    return torch.stack(outputs,-1)


def toy_checks():
    seed_all(4242);head=ToyHead()
    with torch.no_grad():
        # Move away from unit scales so both input/output adaptation are exercised.
        head.first.s.normal_(1,.3);head.second.r.normal_(1,.3);head.second.s.normal_(1,.3)
    x=torch.randn(6,5,dtype=torch.double,requires_grad=True)
    z=head(x);reference=oracle_head(head,x)
    torch.testing.assert_close(z,reference,atol=1e-11,rtol=1e-11)
    params=[x,*head.parameters()];coef=torch.randn_like(z)
    actual=torch.autograd.grad((z*coef).sum(),params,retain_graph=True)
    expected=torch.autograd.grad((reference*coef).sum(),params)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
    losses=[]
    y=torch.tensor([0.,1.,1.,0.,1.,0.],dtype=torch.double)
    for objective in ['member_bce','mixture_bce']:
        logits=(torch.randn(6,4,dtype=torch.double)*2).requires_grad_()
        p=logits.sigmoid();target=y[:,None].expand_as(logits)
        analytic=(target*(-torch.log(p))+(1-target)*(-torch.log1p(-p))).mean() if objective=='member_bce' else (-y*torch.log(p.mean(1))-(1-y)*torch.log1p(-p.mean(1))).mean()
        loss=classification_loss(logits,y,objective)
        ga=torch.autograd.grad(loss,logits,retain_graph=True)[0];gb=torch.autograd.grad(analytic,logits)[0]
        torch.testing.assert_close(loss,analytic,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(ga,gb,atol=1e-12,rtol=1e-12)
        losses.append(dict(objective=objective,loss_error=float(abs(loss-analytic)),gradient_error=float(abs(ga-gb).max())))
    extremes=torch.tensor([[1000.,1000.],[-1000.,-1000.],[1000.,-1000.],[40.,-2.]],dtype=torch.double,requires_grad=True)
    mixed=mixture_logit(extremes)
    torch.testing.assert_close(mixed.sigmoid(),extremes.sigmoid().mean(1),atol=1e-13,rtol=1e-13)
    grad=torch.autograd.grad(F.binary_cross_entropy_with_logits(mixed,torch.tensor([0.,1.,1.,0.],dtype=torch.double)),extremes)[0]
    assert torch.isfinite(mixed).all() and torch.isfinite(grad).all()
    contrast=torch.tensor([[3.,-1.],[-2.,.2]],dtype=torch.double,requires_grad=True);labels=torch.tensor([1.,0.],dtype=torch.double)
    member=classification_loss(contrast,labels,'member_bce');mixture=classification_loss(contrast,labels,'mixture_bce')
    g1=torch.autograd.grad(member,contrast,retain_graph=True)[0];g2=torch.autograd.grad(mixture,contrast)[0]
    assert float(member-mixture)>0 and float(abs(g1-g2).max())>1e-3
    return dict(output_error=float(abs(z-reference).max()),all_parameter_and_input_gradient_error=max(float(abs(a-b).max()) for a,b in zip(actual,expected)),loss_checks=losses,
                extreme_logits_finite=True,extreme_mean_probability_error=float(abs(mixed.sigmoid()-extremes.sigmoid().mean(1)).max()),
                example_loss_gap=float(member-mixture),example_gradient_gap=float(abs(g1-g2).max()))


@torch.no_grad()
def member_deletion_check(a,data,name,count=4):
    a.model.eval();records=query.deletion_check(a,data,name,count)
    for record in records:
        pair=np.asarray(record['pair']);ix=int(np.flatnonzero((data['train_pairs']==pair).all(1))[0])
        keep=np.arange(len(data['train_y']))!=ix
        b=make_adapter({**data,'train_pairs':data['train_pairs'][keep],'train_y':data['train_y'][keep]},name,a.y.device)
        b.model.load_state_dict(a.model.state_dict());b.model.eval()
        r,d=a.pairs['train'][:,ix:ix+1];eli=torch.stack([r,d])
        z1=a.model.member_logits(eli,a.ps,a.qis['train'][ix:ix+1],a.ease['train'][ix:ix+1],a.dense)
        z2=b.model.member_logits(eli,b.ps,(b.ps.qis(r,d)-a.mu)/a.sd,b.ps.ease(r,d),b.dense)
        torch.testing.assert_close(z1,z2,atol=2e-4,rtol=5e-5)
        record['errors']['all_member_logits']=float(abs(z1-z2).max())
        del b
    return records


@torch.no_grad()
def batch_check(a):
    a.model.eval();z=member_logits(a,'val')[:3]
    parts=torch.cat([a.model.member_logits(a.pairs['val'][:,i:i+1],a.ps,a.qis['val'][i:i+1],a.ease['val'][i:i+1],a.dense) for i in range(3)])
    torch.testing.assert_close(z,parts,atol=2e-4,rtol=5e-5)
    return float(abs(z-parts).max())


def main():
    toy=toy_checks();data=load_inner(0)
    seed_all(42);plain=make_adapter(data,'unmasked10','cuda:0');plain.model.eval()
    seed_all(42);identity=make_adapter(data,'identity_rec10','cuda:0');identity.model.eval()
    captured={}
    hook=plain.model.lin1.register_forward_pre_hook(lambda layer,args:captured.update(fusion=args[0].detach()))
    try:
        with torch.no_grad():
            p0=plain.predict('val');p1=identity.predict('val')
            fusion=identity.model.fusion_features(identity.pairs['val'],identity.ps,identity.qis['val'],identity.ease['val'],identity.dense)
    finally:hook.remove()
    np.testing.assert_allclose(p0,p1,atol=2e-5,rtol=2e-5)
    # GPU indexed aggregation is not bitwise reproducible. Use the already
    # established real-model deletion tolerance, while checking inputs exactly.
    for left,right in zip(plain.dense,identity.dense):torch.testing.assert_close(left,right,atol=0,rtol=0)
    torch.testing.assert_close(captured['fusion'],fusion,atol=2e-4,rtol=5e-5)
    fusion_error=float(abs(captured['fusion']-fusion).max())
    with torch.no_grad():
        repeat=identity.model.fusion_features(identity.pairs['val'],identity.ps,identity.qis['val'],identity.ease['val'],identity.dense)
    torch.testing.assert_close(repeat,fusion,atol=2e-4,rtol=5e-5)
    repeat_error=float(abs(repeat-fusion).max())
    records=[]
    for name,cfg in CONFIGS.items():
        seed_all(42);a=make_adapter(data,name,'cuda:0');m=a.model
        for key,value in plain.model.state_dict().items():
            if not key.startswith(('lin1.','lin2.','lin3.')):
                torch.testing.assert_close(value,m.state_dict()[key],atol=0,rtol=0)
        for layer in ['lin1','lin2']:
            torch.testing.assert_close(getattr(plain.model,layer).weight,getattr(m,layer).weight,atol=0,rtol=0)
        opt=make_optimizer(m);m.train();opt.zero_grad();torch.cuda.reset_peak_memory_stats();start=time.time()
        logits=member_logits(a,'train');bce=classification_loss(logits,a.y,cfg['objective'])
        ix=m.reconstruction.masks(a.graph,.5,mask_generator('cuda:0',20261006))
        rec,_=m.reconstruction.loss(m.encoder,a.graph,ix,False);loss=bce+10*rec;loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        norms={}
        for group in ['encoder','ncn_lin','pc','em','lin1','lin2','lin3']:
            norms[group]=sum(float(p.grad.square().sum()) for p in getattr(m,group).parameters() if p.grad is not None)**.5
            assert norms[group]>0
        assert (m.lin3.weight.grad.norm(dim=1)>0).all()
        opt.step();torch.cuda.synchronize();seconds=time.time()-start;peak=torch.cuda.max_memory_allocated()/2**30
        assert int(m.eb.num_batches_tracked)==1
        assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in key.lower() for key in m.state_dict())
        m.eval()
        with torch.no_grad():
            z=member_logits(a,'val');prob=a.predict('val')
            np.testing.assert_allclose(prob,z.sigmoid().mean(1).cpu().numpy(),atol=2e-5,rtol=2e-5)
        records.append(dict(config=name,training_step_seconds=seconds,peak_cuda_gib=peak,gradient_norms=norms,
                            every_output_member_nonzero_gradient=True,batchnorm_updates=1,
                            explicit_deletions=member_deletion_check(a,data,name),batch_member_error=batch_check(a)))
        print('PASS',name,'seconds',seconds,'GiB',peak,flush=True)
        del a,m,opt,loss,logits,rec;torch.cuda.empty_cache()
    seed_all(42);a=make_adapter(data,'independent_rec10','cuda:0')
    seed_all(42);b=make_adapter(data,'mixture_rec10','cuda:0')
    for key,value in a.model.state_dict().items():torch.testing.assert_close(value,b.model.state_dict()[key],atol=0,rtol=0)
    seed_all(42);flipped=make_adapter({**data,'val_y':1-data['val_y']},'independent_rec10','cuda:0')
    a.model.eval();flipped.model.eval()
    with torch.no_grad():pa,pb=a.predict('val'),flipped.predict('val')
    np.testing.assert_allclose(pa,pb,atol=2e-5,rtol=2e-5)
    result=dict(status='passed',toy=toy,records=records,matched_initial_parameters_equal=True,
                original_fusion_input_exactly_equal=True,original_fusion_max_error=fusion_error,repeated_fusion_max_error=repeat_error,
                single_member_original_probability_error=float(abs(p0-p1).max()),
                heldout_label_invariance=True,heldout_probability_error=float(abs(pa-pb).max()),
                query_check_source_sha256=sha256(QUERY_CHECK_SOURCE),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'initial_checks.json').write_text(json.dumps(result,indent=2));print('ALL PASSED',flush=True)

if __name__=='__main__':main()
