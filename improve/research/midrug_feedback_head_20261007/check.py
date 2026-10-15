"""Check the newly trainable head while reusing immutable geometry mathematics."""
import ast,json,time
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,load_inner,seed_all,sha256
from model import CONFIGS,initialize,make_adapter,make_optimizer,trainable,HEAD_PREFIXES,BRANCH_PREFIXES
PREVIOUS=HERE.parent/'midrug_feedback_geometry_20261007'
QUERY_CHECK_SOURCE=HERE.parent/'midrug_degree_encoder_20261006/check.py'

def extract(path,names,namespace):
    tree=ast.parse(path.read_text())
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(body)==len(names)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),namespace)
    return [namespace[n] for n in names]

deletion_check,=extract(QUERY_CHECK_SOURCE,['deletion_check'],dict(torch=torch,np=np,make_adapter=make_adapter))
branch_check,chunk_gradient_check=extract(PREVIOUS/'check.py',['branch_check','chunk_gradient_check'],dict(torch=torch))

def frozen_check(m,state):
    for k,v in state.items():
        if not trainable(k):torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)
    assert all(p.requires_grad==trainable(n) for n,p in m.named_parameters())
    assert all(p.grad is None for n,p in m.named_parameters() if not trainable(n))

def main():
    old=json.loads((PREVIOUS/'archive_manifest.json').read_text());count=0
    for p,h in old['artifact_hashes'].items():assert sha256(PREVIOUS/p)==h;count+=1
    for p,h in old['frozen_dependencies'].items():assert sha256(p)==h;count+=1
    assert sha256(HERE/'modules.py')==sha256(PREVIOUS/'modules.py')
    assert json.loads((PREVIOUS/'math_checks.json').read_text())['status']=='passed'
    records=[]
    for fold in range(3):
        data=load_inner(fold);branch0=None
        for name in CONFIGS:
            seed_all(42);a,path,r=initialize(data,name,fold,'cuda:0');m=a.model;m.eval()
            source=torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True)
            for k,v in source.items():torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)
            branch={k:v.clone() for k,v in m.state_dict().items() if k.startswith(BRANCH_PREFIXES)}
            if branch:
                if branch0 is None:branch0=branch
                for k,v in branch.items():torch.testing.assert_close(v,branch0[k],atol=0,rtol=0)
            with torch.no_grad():prob=a.predict('val')
            with np.load(path.with_name(path.stem+'_inner.npz')) as z:stored=z['probability'].copy()
            np.testing.assert_allclose(prob,stored,atol=2e-5,rtol=2e-5)
            opt=make_optimizer(m);assert not opt.state;losses=[];start=time.time()
            for _ in range(2):
                m.eval()
                if hasattr(m,'feedback'):m.feedback.train()
                opt.zero_grad(set_to_none=True)
                loss=F.binary_cross_entropy_with_logits(a.logits('train'),a.y);assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
                opt.step();losses.append(float(loss));frozen_check(m,source)
            assert all(p.grad.abs().max()>0 for p in m.parameters() if p.requires_grad)
            assert all(not torch.equal(v,m.state_dict()[k]) for k,v in source.items() if k.startswith(HEAD_PREFIXES))
            m.eval();rec=dict(config=name,fold=fold,initial_prediction_error=float(abs(prob-stored).max()),
                losses=losses,seconds_two_updates=time.time()-start,all_frozen_state_exact=True,head_changed=True)
            if branch:rec.update(branch=branch_check(a),chunk=chunk_gradient_check(a))
            b=make_adapter({**data,'val_y':1-data['val_y']},name,'cuda:0');b.model.load_state_dict(m.state_dict());b.model.eval()
            with torch.no_grad():pa,pb=a.predict('val'),b.predict('val')
            np.testing.assert_allclose(pa,pb,atol=2e-5,rtol=2e-5);del b
            rec.update(heldout_flip_error=float(abs(pa-pb).max()),explicit_deletions=deletion_check(a,data,name,count=4))
            records.append(rec);print('PASS',name,fold,flush=True)
            del a,m,opt,source;torch.cuda.empty_cache()
    paths=[HERE/x for x in ['common.py','model.py','modules.py','check.py']]+[QUERY_CHECK_SOURCE,PREVIOUS/'check.py']
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',records=records,
        source_hashes={str(p):sha256(p) for p in paths},prior_archive_hash_checks=count,
        reused_math_checks_sha256=sha256(PREVIOUS/'math_checks.json'),prior_archive_sha256=sha256(PREVIOUS/'archive_manifest.json'),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
