"""Offset alignment, common branch initialization and frozen inference base."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,RETRIEVAL,read_module,load_inner,seed_all,sha256
from model import CONFIGS,initialize,make_adapter,make_optimizer
old=read_module('crossfit_case_checks',RETRIEVAL/'check.py');old.query.make_adapter=make_adapter
deletion_check,memory_check,batch_check=old.deletion_check,old.memory_check,old.batch_check
QUERY_CHECK_SOURCE=old.QUERY_CHECK_SOURCE

def main():
    base=json.loads((HERE/'base_verification.json').read_text());assert base['status']=='passed' and base['new_checkpoint_replays']==18
    records=[]
    for fold in range(3):
        data=load_inner(fold);initial=None
        for name,cfg in CONFIGS.items():
            seed_all(42);a,path,source,offsets=initialize(data,name,fold,'cuda:0');m=a.model;m.eval()
            original={k:v.clone() for k,v in m.state_dict().items() if not k.startswith('case_memory.')}
            if initial is None:initial={k:v.clone() for k,v in m.case_memory.state_dict().items()}
            for k,v in m.case_memory.state_dict().items():torch.testing.assert_close(v,initial[k],atol=0,rtol=0)
            with torch.no_grad():
                delta=m.case_memory(a.pairs['train']);assert (delta==0).all()
                torch.testing.assert_close(a.logits('train'),offsets['in_sample'],atol=2e-4,rtol=5e-5)
                torch.testing.assert_close(a.logits('val'),offsets['val'],atol=2e-4,rtol=5e-5)
            assert (offsets['oof']-offsets['in_sample']).abs().max()>1e-4
            opt=make_optimizer(m);assert not opt.state;offset=offsets[cfg['offset']];losses=[]
            for _ in range(2):
                m.eval();m.case_memory.train();opt.zero_grad(set_to_none=True)
                logit=offset+m.case_memory(a.pairs['train']);loss=F.binary_cross_entropy_with_logits(logit,a.y)
                expected=np.mean(np.logaddexp(0,logit.detach().double().cpu().numpy())-data['train_y']*logit.detach().double().cpu().numpy())
                assert abs(float(loss)-expected)<1e-6
                loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());opt.step();losses.append(float(loss))
                for k,v in original.items():torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)
            assert int(m.eb.num_batches_tracked)==300
            assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('case_memory.'))
            m.eval()
            with torch.no_grad():torch.testing.assert_close(a.logits('train'),offsets['in_sample']+m.case_memory(a.pairs['train']),atol=2e-4,rtol=5e-5)
            records.append(dict(config=name,fold=fold,losses=losses,base_state_unchanged=True,numpy_bce_verified=True,
                memory_check=memory_check(a),batch_max_logit_error=batch_check(a),explicit_deletions=deletion_check(a,data,name,count=4)))
            print('PASS',name,fold,flush=True);del a,m,opt;torch.cuda.empty_cache()
    (HERE/'case_initial_checks.json').write_text(json.dumps(dict(status='passed',records=records,base_verification_sha256=sha256(HERE/'base_verification.json'),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
