"""Warm starts, immutable frozen backbone and original retrieval safety checks."""
import json
import numpy as np
import torch
from common import HERE,PREVIOUS,read_module,load_inner,seed_all,sha256
from model import CONFIGS,initialize,make_adapter,make_optimizer,resumed_sampler,step
old=read_module('warmstart_retrieval_checks',PREVIOUS/'check.py')
old.query.make_adapter=make_adapter
deletion_check,memory_check,batch_check=old.deletion_check,old.memory_check,old.batch_check
QUERY_CHECK_SOURCE=old.QUERY_CHECK_SOURCE

def main():
    records=[]
    for fold in range(3):
        data=load_inner(fold);case_initial=None;sampler_initial=None
        for name,cfg in CONFIGS.items():
            seed_all(42);a,path,record=initialize(data,name,fold,'cuda:0');m=a.model;m.eval()
            source=torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True)
            for k,v in source.items():torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)
            if case_initial is None:case_initial={k:v.clone() for k,v in m.case_memory.state_dict().items()}
            for k,v in m.case_memory.state_dict().items():torch.testing.assert_close(v,case_initial[k],atol=0,rtol=0)
            with np.load(path.with_name(path.stem+'_inner.npz')) as z:stored=z['probability']
            with torch.no_grad():pred=a.predict('val');zero=m.case_memory(a.pairs['train'])
            assert (zero==0).all();np.testing.assert_allclose(pred,stored,atol=2e-5,rtol=2e-5)
            opt=make_optimizer(m);assert not opt.state
            members={id(p) for g in opt.param_groups for p in g['params']}
            assert members=={id(p) for p in m.parameters() if p.requires_grad}
            g=resumed_sampler(a,fold,record['best_epoch'])
            if sampler_initial is None:sampler_initial=g.get_state().clone()
            torch.testing.assert_close(g.get_state(),sampler_initial,atol=0,rtol=0)
            losses=[]
            for ep in range(1,4):
                losses.append(step(a,opt,name,ep,g))
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
                expected_bn=record['best_epoch']+(0 if cfg['freeze_main'] else ep)
                assert int(m.eb.num_batches_tracked)==expected_bn
                if cfg['freeze_main']:
                    for k,v in source.items():torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)
                    assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('case_memory.'))
            assert any(not torch.equal(v,case_initial[k]) for k,v in m.case_memory.state_dict().items())
            if name=='frozen_features':
                assert m.case_memory.label.weight.grad is None and m.case_memory.label.weight not in opt.state
                torch.testing.assert_close(m.case_memory.label.weight,case_initial['label.weight'],atol=0,rtol=0)
            m.eval();records.append(dict(config=name,fold=fold,initial_prediction_error=float(abs(pred-stored).max()),losses=losses,
                source_state_restored=True,empty_optimizer_start=True,frozen_state_exact=cfg['freeze_main'],
                memory_check=memory_check(a),batch_max_logit_error=batch_check(a),explicit_deletions=deletion_check(a,data,name,count=4)))
            print('PASS',name,fold,flush=True);del a,m,opt,source;torch.cuda.empty_cache()
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',records=records,
        source_hashes={str(p):sha256(p) for p in [HERE/'check.py',HERE/'common.py',HERE/'model.py',PREVIOUS/'model.py',PREVIOUS/'check.py',QUERY_CHECK_SOURCE]},
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
