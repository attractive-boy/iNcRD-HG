"""Replay every selected model and independently rebuild deleted-graph evidence."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,corrected_pool
from features import adjacency,transform
from check import explicit
import check_modules

def main():
    plan=json.loads((HERE/'results/plan.json').read_text())
    for file,h in plan['hashes'].items():assert sha256(HERE/file)==h
    for file,h in plan['cache_hashes'].items():assert sha256(HERE/file)==h
    records=[];evidence=[]
    for fold in range(3):
        data=load_inner(fold);A=adjacency(data,'cuda:0')
        ix=np.r_[np.flatnonzero(data['train_y']==1)[[0,77,511]],np.flatnonzero(data['train_y']==0)[[0,77,511]]]
        independent=[]
        for i in ix:
            keep=np.arange(len(data['train_y']))!=i
            deleted={**data,'train_pairs':data['train_pairs'][keep],'train_y':data['train_y'][keep]}
            raw,mass=explicit(adjacency(deleted,'cuda:0'),data['train_pairs'][i])
            independent.append((int(i),raw,mass))
        seed_all(42);a=make_adapter(data,'control',fold,'cuda:0');path=CLEAN/'results'/f'control_f{fold}_s42.json'
        assert sha256(path.with_suffix('.pt'))==plan['controls'][str(fold)]['checkpoint_sha256']
        a.model.load_state_dict(torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True));a.model.eval()
        with torch.no_grad():p=a.predict('val')
        with np.load(path.with_name(path.stem+'_inner.npz')) as z:
            np.testing.assert_allclose(p,z['probability'],atol=2e-5,rtol=2e-5)
            delta=float(abs(p-z['probability']).max())
        records.append(dict(tag=path.stem,reused_control=True,status='passed',max_prediction_error=delta));del a
        for name,cfg in CONFIGS.items():
            path=HERE/'results'/f'{name}_f{fold}_s42.json';r=json.loads(path.read_text())
            assert r['hashes']==plan['hashes'] and r['checkpoint_sha256']==sha256(path.with_suffix('.pt'))
            assert r['no_lncrna'] and not r['outer_validation_evaluated'] and not r['test_evaluated']
            seed_all(42);a=make_adapter(data,name,fold,'cuda:0')
            a.model.load_state_dict(torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True));a.model.eval()
            assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in k.lower() for k in a.model.state_dict())
            with torch.no_grad():p=a.predict('val')
            with np.load(path.with_name(path.stem+'_inner.npz')) as z:
                for key,dkey in [('pairs','val_pairs'),('y','val_y'),('indices','val_indices')]:np.testing.assert_array_equal(z[key],data[dkey])
                np.testing.assert_allclose(p,z['probability'],atol=2e-5,rtol=2e-5);delta=float(abs(p-z['probability']).max())
                assert abs(roc_auc_score(z['y'],z['probability'])-r['inner_auc'])<1e-12
                assert abs(average_precision_score(z['y'],z['probability'])-r['inner_ap'])<1e-12
            curves=json.loads(path.with_name(path.stem+'_curves.json').read_text())
            assert all(abs(x['selection_score']-.5*(x['inner_auc']+x['inner_ap']))<1e-12 for x in curves)
            assert max(curves,key=lambda x:x['selection_score'])['epoch']==r['best_epoch']
            checked=[]
            with torch.no_grad():
                for i,raw,mass in independent:
                    pair=a.pairs['train'][:,i:i+1];w,s=a.model.cache.select(*pair);ew,es=transform(raw,mass,cfg)
                    torch.testing.assert_close(w,ew,atol=0,rtol=0);torch.testing.assert_close(s,es,atol=0,rtol=0)
                    checked.append(dict(index=i,label=int(data['train_y'][i]),cached_evidence_exact=True))
                # Finite learned components, used only for auditable prediction decomposition.
                base,extra=a.model.components(a.pairs['val'],a.ps,a.qis['val'],a.ease['val'],a.dense)
                np.savez_compressed(path.with_name(path.stem+'_components.npz'),base_logit=base.cpu().numpy(),ocn_logit=extra.cpu().numpy())
            checks=check_modules.check(a,count=6)
            records.append(dict(tag=r['tag'],reused_control=False,status='passed',max_prediction_error=delta,
                record_deletion=checked,full_logit_deletion=checks))
            print('PASS',r['tag'],flush=True);del a;torch.cuda.empty_cache()
    result=dict(status='passed',records=records,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'verification.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
