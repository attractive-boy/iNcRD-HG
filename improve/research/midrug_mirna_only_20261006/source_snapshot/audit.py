"""Replay trained miRNA-only checkpoints and check explicit query deletion."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from model import HERE,CONFIGS,load_inner,make_adapter,GroupPairStruct,seed_all,sha256,assert_clean
import check_modules


def main():
    rows=[];original=check_modules.PairStruct
    plan=json.loads((HERE/'results/plan.json').read_text())
    assert plan['groups_source_sha256']==sha256(HERE/'groups.py')
    for fold in range(3):
        data=load_inner(fold)
        for name in CONFIGS:
            path=HERE/'results'/f'{name}_f{fold}_s42.json';r=json.loads(path.read_text())
            assert r['source_sha256']==sha256(HERE/'train.py') and r['model_sha256']==sha256(HERE/'model.py')
            for key,file in [('data_sha256','inner_data.npz'),('groups_sha256','groups.npz'),('graph_sha256','graph.pt')]:assert r[key]==sha256(HERE/file)
            assert r['checkpoint_sha256']==sha256(path.with_suffix('.pt'))
            assert not r['lncrna_used'] and not r['outer_validation_evaluated'] and not r['test_evaluated']
            seed_all(42);a=make_adapter(data,name,'cuda:0');assert_clean(a.graph)
            a.model.load_state_dict(torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True));a.model.eval()
            assert all('lnc' not in key.lower() for key in a.model.state_dict())
            with torch.no_grad():pred=a.predict('val')
            z=np.load(path.with_name(path.stem+'_inner.npz'))
            for k,j in [('pairs','val_pairs'),('y','val_y'),('indices','val_indices')]:np.testing.assert_array_equal(z[k],data[j])
            np.testing.assert_allclose(pred,z['probability'],rtol=2e-5,atol=2e-5)
            assert abs(roc_auc_score(z['y'],z['probability'])-r['inner_auc'])<1e-12
            assert abs(average_precision_score(z['y'],z['probability'])-r['inner_ap'])<1e-12
            curves=json.loads(path.with_name(path.stem+'_curves.json').read_text())
            for c in curves:assert abs(c['selection_score']-.5*(c['inner_auc']+c['inner_ap']))<1e-12
            assert max(curves,key=lambda x:x['selection_score'])['epoch']==r['best_epoch']
            try:
                if name!='control':check_modules.PairStruct=lambda D,p,c:GroupPairStruct(D,p,c,a.ps.evidence,name)
                checks=check_modules.check(a,count=8)
            finally:check_modules.PairStruct=original
            rows.append(dict(tag=r['tag'],status='passed',lncrna_used=False,max_prediction_error=float(np.max(np.abs(pred-z['probability']))),deletion_checks=checks))
            print('PASS',r['tag'],flush=True);del a
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=rows,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
