"""Verify 18 fixed-epoch base models and fold-specific structural contexts."""
import json
import numpy as np
import torch
from common import HERE,Adapter,nested_data,make_graph,seed_all,sha256,Extractor,SG,load_inner
from residual import load_context,SubgraphGNN,predict,CONFIGS

def main():
    plan=json.loads((HERE/'bases/plan.json').read_text());records=[]
    for file,h in plan['source_hashes'].items():assert sha256(file)==h
    for fold in range(3):
        coverage=[]
        for k in [-1,0,1,2,3,4]:
            data,indices=nested_data(fold,k);path=HERE/'bases'/f'base_f{fold}_k{k}.npz';r=json.loads(path.with_suffix('.json').read_text())
            assert sha256(path)==r['cache_sha256'] and sha256(path.with_suffix('.pt'))==r['checkpoint_sha256']
            assert r['epochs']==300 and len(r['train_losses'])==300 and not r['prediction_fold_labels_used_for_selection']
            seed_all(42);a=Adapter(data,make_graph(data),'cuda:0')
            a.model.load_state_dict(torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True));a.model.eval()
            assert set(a.graph.node_types)=={'miRNA','drug'}
            with torch.no_grad():logit=a.logits('val').cpu().numpy()
            with np.load(path) as z:
                np.testing.assert_allclose(logit,z['logit'],atol=2e-5,rtol=2e-5)
                for key,expected in [('prediction_indices',data['val_indices']),('train_indices',data['train_indices']),('prediction_rows',indices)]:np.testing.assert_array_equal(z[key],expected)
                assert not np.intersect1d(z['train_indices'],z['prediction_indices']).size
                ix=np.r_[np.flatnonzero(data['val_y']==1)[[0,11]],np.flatnonzero(data['val_y']==0)[[0,11]]]
                ext=Extractor(data)
                for i in ix:
                    actual=ext.one(*map(int,data['val_pairs'][i]),SG['k'],SG['hops'])
                    for key in actual:np.testing.assert_array_equal(actual[key],z[key][i])
                delta=float(abs(logit-z['logit']).max())
                if k>=0:coverage.extend(z['prediction_rows'].tolist())
            records.append(dict(fold=fold,nested=k,status='passed',max_logit_error=delta,context_queries_checked=len(ix),fixed_epoch=300))
            print('PASS base',fold,k,flush=True);del a;torch.cuda.empty_cache()
        assert sorted(coverage)==list(range(len(load_inner(fold)['train_y'])))
    data,tr,va,offset,vo,hashes=load_context(0,'cuda:0')
    seed_all(42);model=SubgraphGNN(SG,0,0).to('cuda:0');model.eval()
    with torch.no_grad():pred=predict(model,va,vo,CONFIGS['offset_l001'])
    torch.testing.assert_close(pred,vo,rtol=0,atol=0)
    model.train();loss=torch.nn.functional.binary_cross_entropy_with_logits(offset+model(tr,None,None),torch.as_tensor(data['train_y'],device='cuda:0',dtype=torch.float32));loss.backward()
    assert torch.isfinite(loss) and model.head[-1].weight.grad.abs().max()>0
    result=dict(status='passed',records=records,coverage_exactly_once=True,no_overlap=True,zero_residual_equals_base=True,finite_gradient=True,
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'base_verification.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
