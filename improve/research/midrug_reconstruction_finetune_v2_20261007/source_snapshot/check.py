"""Equal starting states, restart semantics, schedules and exact query deletion."""
import json
import numpy as np
import torch
from common import HERE,DEGREE,read_module,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,initialize,make_optimizer,resumed_sampler,reconstruction_weight,step

QUERY_CHECK_SOURCE=DEGREE/'check.py'
query=read_module('reconstruction_finetune_deletion',QUERY_CHECK_SOURCE);query.make_adapter=make_adapter
deletion_check=query.deletion_check


def main():
    schedules={}
    for name in CONFIGS:
        actual=[reconstruction_weight(name,t) for t in [1,50,99,100,101,300]]
        expected={'keep':[10.]*6,'off':[0.]*6,'decay':[9.9,5.,.1,0.,0.,0.]}[name]
        np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=1e-14);schedules[name]=actual
    records=[]
    for fold in range(3):
        data=load_inner(fold);seed_all(42);base,source,record=initialize(data,'keep',fold,'cuda:0')
        initial={k:v.detach().clone() for k,v in base.model.state_dict().items()};base.model.eval()
        with np.load(source.with_name(source.stem+'_inner.npz')) as z:stored=z['probability'].copy()
        with torch.no_grad():replay=base.predict('val')
        np.testing.assert_allclose(replay,stored,atol=2e-5,rtol=2e-5)
        sampler_state=None
        for name in CONFIGS:
            seed_all(42);a,_,_=initialize(data,name,fold,'cuda:0');m=a.model
            for key,value in initial.items():torch.testing.assert_close(value,m.state_dict()[key],atol=0,rtol=0)
            opt=make_optimizer(m);assert not opt.state
            generator=resumed_sampler(a,fold,record['best_epoch'])
            if sampler_state is None:sampler_state=generator.get_state().clone()
            else:torch.testing.assert_close(sampler_state,generator.get_state(),atol=0,rtol=0)
            old={p:p.detach().clone() for group in opt.param_groups for p in group['params']}
            losses=step(a,opt,name,1,generator);assert int(m.eb.num_batches_tracked)==record['best_epoch']+1
            maximum=0.;updated=0;decoder_updated=0
            for group in opt.param_groups:
                for parameter in group['params']:
                    if parameter.grad is None:
                        torch.testing.assert_close(parameter,old[parameter],atol=0,rtol=0);continue
                    assert torch.isfinite(parameter.grad).all();g=parameter.grad+group['weight_decay']*old[parameter]
                    predicted=old[parameter]-group['lr']*g/((.01*g.square()).sqrt()+group['eps'])
                    torch.testing.assert_close(parameter,predicted,atol=1e-7,rtol=1e-6)
                    maximum=max(maximum,float(abs(parameter-predicted).max()));updated+=1
                    assert int(opt.state[parameter]['step'])==1
            for key,value in m.reconstruction.state_dict().items():
                if not torch.equal(value,initial['reconstruction.'+key]):decoder_updated+=1
            if name=='off':assert decoder_updated==0 and not any(p in opt.state for p in m.reconstruction.parameters())
            else:assert decoder_updated>0
            assert updated>0 and all('lnc' not in k.lower() for k in m.state_dict())
            deleted=deletion_check(a,data,name,count=4)
            seed_all(42);flipped=make_adapter({**data,'val_y':1-data['val_y']},name,'cuda:0')
            flipped.model.load_state_dict(m.state_dict());m.eval();flipped.model.eval()
            with torch.no_grad():pa,pb=a.predict('val'),flipped.predict('val')
            np.testing.assert_allclose(pa,pb,atol=2e-5,rtol=2e-5)
            records.append(dict(config=name,fold=fold,source_checkpoint_sha256=record['checkpoint_sha256'],source_epoch=record['best_epoch'],
                initial_prediction_error=float(abs(replay-stored).max()),fresh_rmsprop_update_error=maximum,
                updated_parameter_tensors=updated,changed_decoder_state_tensors=decoder_updated,
                explicit_deletions=deleted,heldout_label_flip_error=float(abs(pa-pb).max()),first_step_losses=losses))
            print('PASS',name,fold,flush=True);del a,m,opt,flipped,old;torch.cuda.empty_cache()
    boundary=[]
    for fold in range(3):
        seed_all(42);a,_,source=initialize(load_inner(fold),'decay',fold,'cuda:0')
        opt=make_optimizer(a.model);generator=resumed_sampler(a,fold,source['best_epoch'])
        for epoch in range(1,100):step(a,opt,'decay',epoch,generator)
        decoder=list(a.model.reconstruction.decoder.parameters())
        assert all(int(opt.state[p]['step'])==99 for p in decoder)
        before={p:p.detach().clone() for p in decoder}
        state={p:{k:v.clone() if torch.is_tensor(v) else v for k,v in opt.state[p].items()} for p in decoder}
        for epoch in [100,101]:
            losses=step(a,opt,'decay',epoch,generator);assert losses['weight']==0
            for p in decoder:
                assert p.grad is None
                torch.testing.assert_close(p,before[p],atol=0,rtol=0)
                for key,value in state[p].items():
                    torch.testing.assert_close(opt.state[p][key],value,atol=0,rtol=0)
            assert int(a.model.eb.num_batches_tracked)==source['best_epoch']+epoch
        boundary.append(dict(fold=fold,decoder_updates=99,main_updates=101,decoder_unchanged_at_steps=[100,101],decoder_grad_none=True))
        print('BOUNDARY PASS',fold,flush=True)
        del a,opt,before,state;torch.cuda.empty_cache()
    result=dict(status='passed',records=records,schedules=schedules,boundary_checks=boundary,all_arms_start_with_identical_weights_buffers_sampler=True,
        optimizer_restart_verified=True,query_check_source_sha256=sha256(QUERY_CHECK_SOURCE),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'initial_checks.json').write_text(json.dumps(result,indent=2));print('ALL PASSED',flush=True)

if __name__=='__main__':main()
