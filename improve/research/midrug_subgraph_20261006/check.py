"""Explicit record-deletion, scope, initial-equivalence and cost checks."""
import json,time
import numpy as np
import torch
from common import HERE,load_inner,seed_all
from extract import Extractor
from model import CONFIGS,make_adapter
import check_modules


def main():
    data=load_inner(0);ext=Extractor(data)
    ix=np.r_[np.flatnonzero(data['train_y']==1)[[0,77,511]],np.flatnonzero(data['train_y']==0)[[0,77,511]]]
    rows=[]
    for k,hops in ((16,1),(32,2)):
        for i in ix:
            keep=np.arange(len(data['train_y']))!=i
            deleted=Extractor({**data,'train_pairs':data['train_pairs'][keep],'train_y':data['train_y'][keep]})
            pair=data['train_pairs'][i]
            before,after=ext.one(*pair,k,hops),deleted.one(*pair,k,hops)
            for key in before:np.testing.assert_array_equal(before[key],after[key])
            assert before['pos'][0,0]==0 and before['neg'][0,0]==0
            rows.append(dict(index=int(i),label=int(data['train_y'][i]),k=k,hops=hops,all_arrays_identical=True))
    seed_all(42);base=make_adapter(data,'control',0,'cuda:0');base.model.eval()
    seed_all(42);a=make_adapter(data,'sg1_signed',0,'cuda:0');a.model.eval()
    assert set(a.graph.node_types)=={'miRNA','drug'}
    assert all('lnc' not in key.lower() for key in a.model.state_dict())
    with torch.no_grad():p0,p1=base.predict('val'),a.predict('val')
    np.testing.assert_allclose(p0,p1,rtol=2e-5,atol=2e-5)
    with torch.no_grad():a.model.subgraph.head[-1].weight.normal_(std=.01)
    checks=check_modules.check(a,count=12)
    a.model.train();opt=torch.optim.RMSprop(a.model.parameters(),lr=1e-4)
    torch.cuda.reset_peak_memory_stats();times=[]
    for step in range(3):
        torch.cuda.synchronize();start=time.time();opt.zero_grad()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(a.logits('train'),a.y)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.model.parameters())
        assert a.model.subgraph.role.weight.grad.abs().max()>0
        opt.step();torch.cuda.synchronize();times.append(time.time()-start)
    result=dict(status='passed',record_deletion=rows,initial_control_max_error=float(np.max(np.abs(p0-p1))),
        final_logit_deletion=checks,finite_nonzero_subgraph_gradients=True,no_lncrna=True,
        full_batch_step_seconds=times,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
        outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'initial_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
