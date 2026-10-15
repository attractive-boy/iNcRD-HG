"""Actual graph/record deletion with the same weights and independent extraction."""
import ast,json,time
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,load_inner,seed_all,sha256
from model import CONFIGS,initialize,make_adapter,make_optimizer,Extractor,OLD,EXTRACT_SOURCE
QUERY_CHECK_SOURCE=HERE.parent/'midrug_degree_encoder_20261006/check.py'
tree=ast.parse(QUERY_CHECK_SOURCE.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='deletion_check')
ns=dict(torch=torch,np=np,make_adapter=make_adapter)
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(QUERY_CHECK_SOURCE),'exec'),ns)
deletion_check=ns['deletion_check']

@torch.no_grad()
def branch_check(a,data,name):
    m=a.model;m.eval();ix=np.r_[np.flatnonzero(data['train_y']==1)[[0,-1]],np.flatnonzero(data['train_y']==0)[[0,-1]]]
    q=a.pairs['train'][:,ix];batch=m.cache.select(*q);branch=m.subgraph
    def forward(b):return branch({**b,'neg':torch.zeros_like(b['pos'])} if name=='mpnn' else b,*a.dense[:2])
    out=forward(batch);alone=torch.cat([forward({k:v[j:j+1] for k,v in batch.items()}) for j in range(4)])
    torch.testing.assert_close(out,alone,atol=2e-5,rtol=2e-5)
    errors=[]
    for j,index in enumerate(ix):
        keep=np.arange(len(data['train_y']))!=index
        other={**data,'train_pairs':data['train_pairs'][keep],'train_y':data['train_y'][keep]}
        raw=Extractor(other).one(*data['train_pairs'][index],16,1)
        for key in batch:np.testing.assert_array_equal(raw[key],batch[key][j].cpu().numpy())
        converted={k:torch.as_tensor(raw[k][None],device=a.y.device,dtype=batch[k].dtype) for k in batch}
        actual=forward(converted);torch.testing.assert_close(actual,out[j:j+1],atol=2e-5,rtol=2e-5)
        errors.append(float((actual-out[j:j+1]).abs().max()))
    # Reordering within each type preserves features, roots, edges and output.
    rp=torch.tensor([0]+list(range(15,0,-1)),device=a.y.device);dp=rp.clone();whole=torch.cat([rp,16+dp])
    perm=dict(rid=batch['rid'][:,rp],did=batch['did'][:,dp],features=batch['features'][:,whole],pos=batch['pos'][:,rp][:,:,dp])
    shifted=forward(perm);torch.testing.assert_close(out,shifted,atol=2e-5,rtol=2e-5)
    return dict(batch_error=float((out-alone).abs().max()),record_deletion_errors=errors,all_cached_arrays_rebuilt_equal=True,permutation_error=float((out-shifted).abs().max()))

def main():
    math=json.loads((HERE/'math_checks.json').read_text());assert math['status']=='passed'
    for p,h in math['source_hashes'].items():assert sha256(p)==h
    rows=[]
    for fold in range(3):
        data=load_inner(fold);kernel_initial=None
        for name in CONFIGS:
            seed_all(42);a,path,r=initialize(data,name,fold,'cuda:0');m=a.model;m.eval()
            source=torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True)
            initial={k:v.clone() for k,v in m.subgraph.state_dict().items()}
            if name!='mpnn':
                if kernel_initial is None:kernel_initial=initial
                for k,v in initial.items():torch.testing.assert_close(v,kernel_initial[k],atol=0,rtol=0)
            with torch.no_grad():prob=a.predict('val')
            with np.load(path.with_name(path.stem+'_inner.npz')) as z:stored=z['probability'].copy()
            np.testing.assert_allclose(prob,stored,atol=2e-5,rtol=2e-5)
            opt=make_optimizer(m);assert not opt.state;start=time.time();losses=[]
            for _ in range(2):
                m.eval();m.subgraph.train();opt.zero_grad(set_to_none=True)
                loss=F.binary_cross_entropy_with_logits(a.logits('train'),a.y);assert torch.isfinite(loss);loss.backward()
                for n,p in m.named_parameters():
                    if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all()
                    else:assert p.grad is None
                opt.step();losses.append(float(loss))
                for k,v in source.items():torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)
            assert all(p.grad.abs().max()>0 for p in m.parameters() if p.requires_grad)
            if name in ['fixed','no_edges']:torch.testing.assert_close(m.subgraph.edge_raw,initial['edge_raw'],atol=0,rtol=0)
            m.eval();b=make_adapter({**data,'val_y':1-data['val_y']},name,'cuda:0');b.model.load_state_dict(m.state_dict());b.model.eval()
            with torch.no_grad():pa,pb=a.predict('val'),b.predict('val')
            np.testing.assert_allclose(pa,pb,atol=2e-5,rtol=2e-5);del b
            rows.append(dict(config=name,fold=fold,initial_prediction_error=float(abs(prob-stored).max()),losses=losses,seconds_two_updates=time.time()-start,
                frozen_original_exact=True,all_trainable_gradients_finite_nonzero=True,heldout_flip_error=float(abs(pa-pb).max()),
                branch=branch_check(a,data,name),explicit_deletions=deletion_check(a,data,name,count=4)))
            print('PASS',name,fold,flush=True);del a,m,opt,source;torch.cuda.empty_cache()
    files=[HERE/p for p in ['check.py','model.py','common.py']]+[QUERY_CHECK_SOURCE,OLD/'model.py',EXTRACT_SOURCE]
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',records=rows,source_hashes={str(p):sha256(p) for p in files},
        math_checks_sha256=sha256(HERE/'math_checks.json'),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
