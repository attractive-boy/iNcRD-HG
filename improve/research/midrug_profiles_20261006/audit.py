"""Trained-checkpoint profile replay, selection audit and deletion equivalence."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from model import HERE,CONFIGS,make_adapter,load_fold,seed_all,sha256
from train import anchor_metrics
from check import check_profiles
import check_modules


def main():
    data=load_fold('MiDrug',0);ba,bp=anchor_metrics(data)
    rows=[]
    for name,cfg in CONFIGS.items():
        path=HERE/'results'/f'{name}_s42.json';rec=json.loads(path.read_text())
        assert rec['source_sha256']==sha256(HERE/'train.py') and rec['model_sha256']==sha256(HERE/'model.py')
        assert rec['checkpoint_sha256']==sha256(path.with_suffix('.pt'))
        seed_all(42);a=make_adapter(data,cfg,'cuda:0')
        a.model.load_state_dict(torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True));a.model.eval()
        with torch.no_grad():pred=a.predict('val')
        z=np.load(path.with_name(path.stem+'_val.npz'))
        np.testing.assert_array_equal(z['pairs'],data['val_pairs']);np.testing.assert_array_equal(z['y'],data['val_y'])
        np.testing.assert_allclose(pred,z['probability'],rtol=2e-5,atol=2e-5)
        assert abs(roc_auc_score(z['y'],z['probability'])-rec['val_auc'])<1e-12
        assert abs(average_precision_score(z['y'],z['probability'])-rec['val_ap'])<1e-12
        curves=json.loads(path.with_name(path.stem+'_curves.json').read_text())
        for r in curves:assert abs(r['selection_score']-min(r['val_auc']-ba,r['val_ap']-bp))<1e-12
        assert max(curves,key=lambda x:x['selection_score'])['epoch']==rec['best_epoch']
        branch=check_profiles(a,data)
        full=check_modules.check(a,count=12)
        rows.append(dict(tag=rec['tag'],status='passed',profile_deletion=branch,final_logit_deletion=full,
            max_replay_error=float(np.max(np.abs(pred-z['probability'])))))
        print('PASS',rec['tag'],flush=True)
        del a
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=rows,test_evaluated=False),indent=2))

if __name__=='__main__':main()
