"""Real graph gradients, source freezing, chunking, and full-model deletion."""
import ast,json,time
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,load_inner,seed_all,sha256
from model import CONFIGS,initialize,make_adapter,make_optimizer
QUERY_CHECK_SOURCE=HERE.parent/'midrug_degree_encoder_20261006/check.py'
# Reuse exactly the standalone physical deletion checker, avoiding its unrelated
# CLI imports and training entry point. Its source hash is frozen in the plan.
tree=ast.parse(QUERY_CHECK_SOURCE.read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='deletion_check')
namespace=dict(torch=torch,np=np,make_adapter=make_adapter)
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(QUERY_CHECK_SOURCE),'exec'),namespace)
deletion_check=namespace['deletion_check']


@torch.no_grad()
def branch_check(a):
    f=a.model.feedback;f.eval();pairs=a.pairs['train'][:,[0,1,-1]]
    actual=f(*a.dense[:2],pairs)
    reference=torch.cat([f.explicit(*a.dense[:2],q) for q in pairs.T])
    torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-4)
    alone=torch.cat([f(*a.dense[:2],q[:,None]) for q in pairs.T])
    torch.testing.assert_close(actual,alone,atol=2e-5,rtol=2e-4)
    # Check mixed known/absent graph membership is unrelated to query position.
    shuffled=f(*a.dense[:2],pairs[:,[2,0,1]])
    torch.testing.assert_close(shuffled,actual[[2,0,1]],atol=2e-5,rtol=2e-4)
    return dict(explicit_two_step_error=float((actual-reference).abs().max()),batch_error=float((actual-alone).abs().max()))


def chunk_gradient_check(a):
    f=a.model.feedback;f.train();pairs=a.pairs['train'][:,:70];pars=list(f.parameters())
    grads=[];outs=[]
    for size in [256,17]:
        f.chunk_size=size;out=f(*a.dense[:2],pairs)
        grad=torch.autograd.grad(out.square().mean(),pars)
        grads.append(grad);outs.append(out.detach())
    f.chunk_size=256;f.eval()
    torch.testing.assert_close(outs[0],outs[1],atol=2e-5,rtol=2e-4)
    for u,v in zip(*grads):torch.testing.assert_close(u,v,atol=2e-5,rtol=2e-4)
    return dict(output_error=float((outs[0]-outs[1]).abs().max()),gradient_error=max(float((u-v).abs().max()) for u,v in zip(*grads)))


def main():
    math=json.loads((HERE/'math_checks.json').read_text());assert math['status']=='passed'
    for p,h in math['source_hashes'].items():assert sha256(p)==h
    records=[]
    for fold in range(3):
        data=load_inner(fold);first=None
        for name in CONFIGS:
            seed_all(42);a,path,r=initialize(data,name,fold,'cuda:0');m=a.model;m.eval()
            source=torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True)
            for k,v in source.items():torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)
            branch={k:v.clone() for k,v in m.state_dict().items() if k.startswith(('feedback.','feedback_projection.'))}
            if first is None:first=branch
            for k,v in branch.items():torch.testing.assert_close(v,first[k],atol=0,rtol=0)
            with torch.no_grad():prob=a.predict('val')
            with np.load(path.with_name(path.stem+'_inner.npz')) as z:stored=z['probability']
            np.testing.assert_allclose(prob,stored,atol=2e-5,rtol=2e-5)
            opt=make_optimizer(m);assert not opt.state
            torch.cuda.synchronize();start=time.time();torch.cuda.reset_peak_memory_stats();losses=[]
            for _ in range(2):
                m.eval();m.feedback.train();opt.zero_grad(set_to_none=True)
                loss=F.binary_cross_entropy_with_logits(a.logits('train'),a.y);assert torch.isfinite(loss)
                loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
                opt.step();losses.append(float(loss))
                for k,v in source.items():torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)
                assert int(m.eb.num_batches_tracked)==r['best_epoch']
            assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith(('feedback.','feedback_projection.')))
            assert all(p.grad is not None and p.grad.abs().max()>0 for p in m.feedback.parameters())
            torch.cuda.synchronize();elapsed=time.time()-start;peak=torch.cuda.max_memory_allocated()/2**30;m.eval()
            check=branch_check(a);chunk=chunk_gradient_check(a)
            # Flip held-out labels while preserving the trained weights and data.
            b=make_adapter({**data,'val_y':1-data['val_y']},name,'cuda:0');b.model.load_state_dict(m.state_dict());b.model.eval()
            with torch.no_grad():pa,pb=a.predict('val'),b.predict('val')
            np.testing.assert_allclose(pa,pb,atol=2e-5,rtol=2e-5);del b
            records.append(dict(config=name,fold=fold,initial_prediction_error=float(abs(prob-stored).max()),losses=losses,
                source_state_frozen=True,all_branch_gradients_finite_nonzero=True,seconds_two_updates=elapsed,peak_gib=peak,
                neighborhood_shape=list(m.feedback.neighbors.shape),branch=check,chunk=chunk,
                heldout_flip_error=float(abs(pa-pb).max()),explicit_deletions=deletion_check(a,data,name,count=4)))
            print('PASS',name,fold,'seconds',round(elapsed,2),flush=True)
            del a,m,opt,source;torch.cuda.empty_cache()
    out=dict(status='passed',records=records,source_hashes={str(p):sha256(p) for p in [HERE/'check.py',HERE/'modules.py',HERE/'model.py',HERE/'common.py',QUERY_CHECK_SOURCE]},
        math_checks_sha256=sha256(HERE/'math_checks.json'),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'initial_checks.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__':main()
