"""Visible pooling, masked input isolation, EMA recurrence and full inference."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,DEGREE,read_module,load_inner,seed_all,sha256
from model import CONFIGS,encode,normalized_target,cosine_loss,make_adapter,make_optimizer,step,mask_generator,load_masks

QUERY_CHECK_SOURCE=DEGREE/'check.py'
query=read_module('structural_teacher_deletion',QUERY_CHECK_SOURCE);query.make_adapter=make_adapter
deletion_check=query.deletion_check


def context_check(a,masked):
    m=a.model;aux=m.context_teacher;m.eval()
    inputs={n:a.graph[n].x.detach().clone().requires_grad_() for n in ['miRNA','drug']}
    actual=aux.context(m.encoder,a.graph,masked,inputs)
    objective=sum(x.square().mean() for x in actual.values())
    grads=torch.autograd.grad(objective,list(inputs.values()))
    for n,g in zip(inputs,grads):assert torch.isfinite(g).all() and (g[masked[n]]==0).all()
    with torch.no_grad():
        changed={n:torch.where(masked[n][:,None],x+12345.,x) for n,x in inputs.items()}
        alternative=aux.context(m.encoder,a.graph,masked,changed)
        damaged={n:torch.where(masked[n][:,None],aux.tokens[n],x) for n,x in inputs.items()}
        states=encode(m.encoder,m.encoder.original,damaged)
        states={n:F.layer_norm(x,(x.shape[1],),eps=1e-5).double().cpu().numpy() for n,x in states.items()}
        errors=[];masked_errors=[]
        for n in inputs:
            torch.testing.assert_close(actual[n],alternative[n],atol=2e-4,rtol=5e-5)
            masked_errors.append(float(abs(actual[n]-alternative[n]).max()))
            values=[];descriptors=[]
            for e in sorted([e for e in a.graph.edge_types if e[2]==n],key=lambda e:e[0]!=n):
                adj=m.encoder.adj[e].to_dense().double().cpu().numpy();visible=(~masked[e[0]]).cpu().numpy()
                degree=adj.sum(1,keepdims=True);count=adj[:,visible].sum(1,keepdims=True)
                values.append((adj[:,visible]@states[e[0]][visible])/np.maximum(count,1))
                descriptors.extend([np.log1p(degree),np.log1p(count)])
            expected=np.concatenate(values+descriptors,1)
            np.testing.assert_allclose(actual[n].detach().cpu(),expected,atol=2e-4,rtol=5e-5)
            errors.append(float(abs(actual[n].detach().cpu().numpy()-expected).max()))
    return dict(dense_visible_pool_error=max(errors),masked_attribute_error=max(masked_errors),masked_input_gradient_exact_zero=True)


def recurrence_check(a,before):
    source=dict(a.model.encoder.original.named_parameters());teacher=dict(a.model.context_teacher.teacher.named_parameters());momentum=a.context_config['momentum'];errors=[]
    for name,t in teacher.items():
        expected=before[name].double()*momentum+source[name].detach().double()*(1-momentum)
        torch.testing.assert_close(t.double(),expected,atol=2e-7,rtol=2e-6)
        if momentum==0:torch.testing.assert_close(t,source[name],atol=0,rtol=0)
        errors.append(float(abs(t.double()-expected).max()));assert not t.requires_grad and t.grad is None
    return max(errors)


def main():
    manifest=json.loads((HERE/'mask_manifest.json').read_text());assert sha256(HERE/'masks.npz')==manifest['masks_sha256']
    assert manifest['paired_node_exposures_matched'] and not manifest['uses_association_labels']
    gen=torch.Generator().manual_seed(123)
    pred=torch.randn(7,5,generator=gen,dtype=torch.float64,requires_grad=True)
    raw=torch.randn(7,5,generator=gen,dtype=torch.float64);target=normalized_target(raw)
    expected_target=raw.numpy()-raw.numpy().mean(0);expected_target/=np.linalg.norm(expected_target,axis=1,keepdims=True)
    np.testing.assert_allclose(target,expected_target,atol=1e-13,rtol=1e-13)
    expected=(1-(pred.detach().numpy()/np.linalg.norm(pred.detach().numpy(),axis=1,keepdims=True)*expected_target).sum(1)).mean()
    np.testing.assert_allclose(float(cosine_loss(pred,target)),expected,atol=1e-13,rtol=1e-13)
    assert torch.autograd.gradcheck(lambda p:cosine_loss(p,target),(pred,),eps=1e-6,atol=1e-6,rtol=1e-5)
    zero=cosine_loss(pred,torch.zeros_like(target));assert zero==0
    assert (torch.autograd.grad(zero,pred)[0]==0).all()
    records=[]
    for fold in range(3):
        data=load_inner(fold);schedules=load_masks(fold,'cuda:0')
        for n in ['miRNA','drug']:
            b,r=schedules['block'][n],schedules['random'][n]
            assert (b.sum(1)==round(.5*b.shape[1])).all() and (r.sum(1)==b.sum(1)).all()
            torch.testing.assert_close(b.reshape(400,2,-1).sum(1),r.reshape(400,2,-1).sum(1),atol=0,rtol=0)
        seed_all(42);base=make_adapter(data,'unmasked10','cuda:0');base.model.eval()
        initial={k:v.detach().clone() for k,v in base.model.state_dict().items()}
        with torch.no_grad():p0=base.predict('val')
        aux_initial=None
        for name in CONFIGS:
            seed_all(42);a=make_adapter(data,name,'cuda:0');a.mask_schedule=schedules;a.model.eval()
            student=dict(a.model.encoder.original.named_parameters())
            for key,value in a.model.context_teacher.teacher.named_parameters():
                torch.testing.assert_close(value,student[key],atol=0,rtol=0);assert not value.requires_grad
            for key,value in initial.items():torch.testing.assert_close(value,a.model.state_dict()[key],atol=0,rtol=0)
            if aux_initial is None:aux_initial={k:v.detach().clone() for k,v in a.model.context_teacher.state_dict().items()}
            for key,value in aux_initial.items():torch.testing.assert_close(value,a.model.context_teacher.state_dict()[key],atol=0,rtol=0)
            with torch.no_grad():p1=a.predict('val')
            np.testing.assert_allclose(p0,p1,atol=2e-5,rtol=2e-5)
            opt=make_optimizer(a.model);generator=mask_generator('cuda:0',20261006+fold);errors=[];losses=[]
            # Two real mechanism-check updates, not a formal training trajectory.
            for count,epoch in enumerate([1,31],1):
                before={k:v.detach().clone() for k,v in a.model.context_teacher.teacher.named_parameters()}
                head_before={k:v.detach().clone() for k,v in a.model.context_teacher.named_parameters() if v.requires_grad}
                losses.append(step(a,opt,epoch,generator));errors.append(recurrence_check(a,before))
                assert int(a.model.eb.num_batches_tracked)==count and int(a.model.context_teacher.teacher_updates)==count
                if epoch==1:
                    for k,v in a.model.context_teacher.named_parameters():
                        if v.requires_grad:torch.testing.assert_close(v,head_before[k],atol=0,rtol=0);assert v.grad is None
                else:assert any(v.grad is not None and v.grad.abs().max()>0 for v in a.model.context_teacher.predictor.parameters())
            seed_all(42);other=make_adapter({**data,'val_y':1-data['val_y']},name,'cuda:0');other.model.load_state_dict(a.model.state_dict());a.model.eval();other.model.eval()
            mask={n:x[30] for n,x in schedules[a.context_config['mask']].items()}
            with torch.no_grad():
                pa,pb=a.predict('val'),other.predict('val')
                la,_=a.model.context_teacher.loss(a.model.encoder,a.graph,mask)
                lb,_=other.model.context_teacher.loss(other.model.encoder,other.graph,mask)
            np.testing.assert_allclose(pa,pb,atol=2e-5,rtol=2e-5);torch.testing.assert_close(la,lb,atol=2e-5,rtol=2e-5)
            records.append(dict(config=name,fold=fold,initial_error=float(abs(p0-p1).max()),losses=losses,ema_max_error=max(errors),
                heldout_flip_error=float(abs(pa-pb).max()),context_check=context_check(a,mask),explicit_deletions=deletion_check(a,data,name,count=4)))
            print('PASS',name,fold,flush=True);del a,other,opt;torch.cuda.empty_cache()
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',records=records,numpy_cosine_and_target=True,float64_gradcheck=True,
        paired_node_exposures_matched=True,teacher_stop_gradient=True,disabled_auxiliary_parameters_unchanged=True,
        query_check_source_sha256=sha256(QUERY_CHECK_SOURCE),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
    print('ALL PASSED',flush=True)


if __name__=='__main__':main()
