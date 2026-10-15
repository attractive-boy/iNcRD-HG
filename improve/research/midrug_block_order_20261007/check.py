"""Independent scalar RMSprop oracle and actual M123 update/BN checks."""
import ast,json
import numpy as np
import torch
from common import HERE,load_inner,seed_all
from model import CONFIGS,make_adapter,make_optimizers,mask_generator,objective,update_blocks,train_epoch,batch_statistics_without_tracking
QUERY_SOURCE=HERE.parent/'midrug_degree_encoder_20261006/check.py'
tree=ast.parse(QUERY_SOURCE.read_text());body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='deletion_check']
ns=dict(np=np,torch=torch,make_adapter=make_adapter);exec(compile(ast.Module(body=body,type_ignores=[]),str(QUERY_SOURCE),'exec'),ns);deletion_check=ns['deletion_check']

def scalar_check():
    records=[]
    for order in CONFIGS:
        pars={n:torch.nn.Parameter(torch.tensor(v,dtype=torch.double)) for n,v in dict(r=.4,c=-.7,d=.2).items()}
        opts=dict(C=torch.optim.RMSprop([pars['c']],lr=1e-4,alpha=.99),R=torch.optim.RMSprop([
            dict(params=[pars['r']],lr=1e-4),dict(params=[pars['d']],lr=1e-4,weight_decay=1e-4)],alpha=.99))
        values=dict(r=.4,c=-.7,d=.2);mom={n:0. for n in values};calls=[]
        def grad(v):
            er=v['r']*v['c']-.7;ed=v['r']*v['d']+.3
            return dict(r=er*v['c']+3*ed*v['d'],c=er*v['r'],d=3*ed*v['r'])
        def step(names,g):
            for n in names:
                gg=g[n]+(1e-4*values[n] if n=='d' else 0.)
                mom[n]=.99*mom[n]+.01*gg**2;values[n]-=1e-4*gg/(np.sqrt(mom[n])+1e-8)
        def closure(which):
            calls.append(which);r,c,d=pars['r'],pars['c'],pars['d']
            loss=.5*(r*c-.7)**2+1.5*(r*d+.3)**2
            return loss,dict(value=float(loss.detach()))
        errors=[]
        for epoch in range(1,6):
            g=grad(values)
            if order=='joint':step(['c','r','d'],g)
            elif order=='consumer_first':step(['c'],g);step(['r','d'],grad(values))
            else:step(['r','d'],g);step(['c'],grad(values))
            update_blocks(list(pars.values()),opts,closure,order)
            for n,p in pars.items():
                errors.append(abs(float(p)-values[n]));assert errors[-1]<1e-14
                state=opts['C' if n=='c' else 'R'].state[p]
                assert int(state['step'])==epoch and abs(float(state['square_avg'])-mom[n])<1e-14
        assert calls==[0,1]*5;records.append(dict(order=order,max_parameter_error=max(errors),closure_calls=len(calls),optimizer_steps_per_parameter=5))
    m=torch.nn.Module();m.eb=torch.nn.BatchNorm1d(3);m.train();x=torch.tensor([[1.,2.,3.],[2.,4.,1.],[4.,1.,3.]])
    a=m.eb(x);before={k:v.clone() for k,v in m.eb.state_dict().items()}
    with batch_statistics_without_tracking(m):b=m.eb(x)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    for k,v in before.items():torch.testing.assert_close(v,m.eb.state_dict()[k],atol=0,rtol=0)
    try:
        with batch_statistics_without_tracking(m):raise ValueError('intentional restoration probe')
    except ValueError:pass
    assert m.eb.track_running_stats
    return dict(records=records,bn_same_batch_output=True,bn_second_pass_state_unchanged=True,bn_exception_restored=True)

def main():
    scalar=scalar_check();data=load_inner(0);records=[]
    # Same actual graph gradient removes independent CUDA reduction variability
    # from the claim that splitting the optimizer preserves one joint update.
    seed_all(42);u=make_adapter(data,'joint','cuda:0');seed_all(42);w=make_adapter(data,'joint','cuda:0')
    ou=make_optimizers(u.model);ow=torch.optim.RMSprop([
        dict(params=[p for n,p in w.model.named_parameters() if not n.startswith('reconstruction.')],lr=1e-4),
        dict(params=w.model.reconstruction.parameters(),lr=1e-4,weight_decay=1e-4)],alpha=.99)
    u.model.train();ix=u.model.reconstruction.masks(u.graph,.5,mask_generator('cuda:0',20261006))
    value,_=objective(u,ix,.2);value.backward()
    for (n,p),(nn,q) in zip(u.model.named_parameters(),w.model.named_parameters()):
        assert n==nn;q.grad=None if p.grad is None else p.grad.detach().clone()
    ou['C'].step();ou['R'].step();ow.step();same_gradient_errors=[]
    for (n,p),(nn,q) in zip(u.model.named_parameters(),w.model.named_parameters()):
        assert n==nn;torch.testing.assert_close(p,q,atol=0,rtol=0);same_gradient_errors.append(float(abs(p-q).max()))
    del u,w,ou,ow;torch.cuda.empty_cache()
    # Compare new joint control with the original single optimizer/one-pass update.
    seed_all(42);a=make_adapter(data,'joint','cuda:0');seed_all(42);b=make_adapter(data,'joint','cuda:0')
    oa=make_optimizers(a.model);ob=torch.optim.RMSprop([
        dict(params=[p for n,p in b.model.named_parameters() if not n.startswith('reconstruction.')],lr=1e-4),
        dict(params=b.model.reconstruction.parameters(),lr=1e-4,weight_decay=1e-4)],alpha=.99)
    ga=mask_generator('cuda:0',20261006);gb=mask_generator('cuda:0',20261006);joint_errors=[]
    for epoch in range(1,4):
        train_epoch(a,oa,CONFIGS['joint'],epoch,ga);b.model.train();ob.zero_grad(set_to_none=True)
        indices=b.model.reconstruction.masks(b.graph,.5,gb);loss,_=objective(b,indices,10*min(1.,epoch/50.));loss.backward();ob.step()
        assert int(a.model.eb.num_batches_tracked)==int(b.model.eb.num_batches_tracked)==epoch
        torch.testing.assert_close(ga.get_state(),gb.get_state(),atol=0,rtol=0)
        for k,v in a.model.state_dict().items():
            w=b.model.state_dict()[k];torch.testing.assert_close(v,w,atol=3e-5,rtol=1e-4)
            if v.is_floating_point():joint_errors.append(float(abs(v-w).max()))
    del a,b,oa,ob;torch.cuda.empty_cache()
    for name,cfg in CONFIGS.items():
        seed_all(42);a=make_adapter(data,name,'cuda:0');opts=make_optimizers(a.model);g=mask_generator('cuda:0',20261006)
        assert [n for n,m in a.model.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]==['eb']
        stats=[train_epoch(a,opts,cfg,epoch,g) for epoch in [1,2]]
        for o in opts.values():
            for state in o.state.values():
                assert int(state['step'])==2
                assert all(not torch.is_tensor(v) or torch.isfinite(v).all() for v in state.values())
        assert int(a.model.eb.num_batches_tracked)==2 and a.model.eb.track_running_stats
        a.model.eval();deleted=deletion_check(a,data,name,count=4)
        b=make_adapter({**data,'val_y':1-data['val_y']},name,'cuda:0');b.model.load_state_dict(a.model.state_dict());b.model.eval()
        with torch.no_grad():pa,pb=a.predict('val'),b.predict('val')
        np.testing.assert_allclose(pa,pb,atol=2e-5,rtol=2e-5)
        records.append(dict(config=name,steps=2,statistics=stats,physical_deletions=deleted,
            optimizer_state_counts={k:len(o.state) for k,o in opts.items()},heldout_flip_error=float(abs(pa-pb).max())))
        print('PASS',name,flush=True);del a,b,opts;torch.cuda.empty_cache()
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',scalar_oracle=scalar,
        joint_same_actual_gradient_update_error=max(same_gradient_errors),joint_independent_cuda_trajectory_parameter_max_error=max(joint_errors),
        records=records,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
if __name__=='__main__':main()
