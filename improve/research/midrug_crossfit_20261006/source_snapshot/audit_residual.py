"""Replay nine residual models, selected metrics, epochs and nested graph inputs."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,SG,SubgraphGNN,Extractor,nested_data,seed_all,sha256
from residual import CONFIGS,load_context,predict,KEYS

def main():
    plan=json.loads((HERE/'results/plan.json').read_text())
    for p,h in plan['source_hashes'].items():assert sha256(p)==h
    records=[]
    for fold in range(3):
        outer,tr,va,offset,vo,hashes=load_context(fold,'cuda:0');samples=[]
        for k in range(5):
            data,indices=nested_data(fold,k);ext=Extractor(data)
            for lab in (1,0):
                j=int(np.flatnonzero(data['val_y']==lab)[0]);i=int(indices[j]);pair=data['val_pairs'][j]
                actual=ext.one(*map(int,pair),SG['k'],SG['hops'])
                batch={key:torch.as_tensor(x[None],device='cuda:0',dtype=tr[key].dtype) for key,x in actual.items()}
                for key in KEYS:torch.testing.assert_close(tr[key][i:i+1],batch[key],atol=0,rtol=0)
                samples.append((i,batch))
        for name,cfg in CONFIGS.items():
            path=HERE/'results'/f'{name}_f{fold}_s42.json';r=json.loads(path.read_text())
            assert r['checkpoint_sha256']==sha256(path.with_suffix('.pt')) and r['cache_hashes']==hashes
            assert r['source_sha256']==sha256(HERE/'residual.py')
            seed_all(42);model=SubgraphGNN(SG,0,0).to('cuda:0')
            model.load_state_dict(torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True));model.eval()
            with torch.no_grad():p=predict(model,va,vo,cfg).sigmoid().cpu().numpy()
            with np.load(path.with_name(path.stem+'_inner.npz')) as z:
                np.testing.assert_array_equal(z['pairs'],outer['val_pairs']);np.testing.assert_array_equal(z['y'],outer['val_y'])
                np.testing.assert_allclose(z['probability'],p,atol=2e-5,rtol=2e-5)
                assert abs(roc_auc_score(z['y'],z['probability'])-r['inner_auc'])<1e-12
                assert abs(average_precision_score(z['y'],z['probability'])-r['inner_ap'])<1e-12
                delta=float(abs(p-z['probability']).max())
            curves=json.loads(path.with_name(path.stem+'_curves.json').read_text())
            assert all(abs(x['selection_score']-.5*(x['inner_auc']+x['inner_ap']))<1e-12 for x in curves)
            assert max(curves,key=lambda x:x['selection_score'])['epoch']==r['best_epoch']
            with torch.no_grad():
                for i,batch in samples:
                    actual=model(batch,None,None);expected=model({key:x[i:i+1] for key,x in tr.items()},None,None)
                    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            records.append(dict(tag=r['tag'],status='passed',max_prediction_error=delta,nested_context_checks=len(samples)))
            print('PASS',r['tag'],flush=True);del model
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=records,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
