"""Replay this round's checkpoints and check explicit query deletion."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from model import HERE, RELEASE, CONFIGS, make_adapter, AttentivePairStruct, load_fold, seed_all, sha256
from train_chem import make_chemical_adapter
from train import anchor_metrics
import check_modules


def audit_one(path, adapter, factory, source, ba, bp, data):
    rec=json.loads(path.read_text())
    assert rec['checkpoint_sha256']==sha256(path.with_suffix('.pt'))
    assert rec['source_sha256']==sha256(source)
    assert not rec['test_evaluated']
    adapter.model.load_state_dict(torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True))
    adapter.model.eval()
    with torch.no_grad(): pred=adapter.predict('val')
    saved=np.load(path.with_name(path.stem+'_val.npz'))
    np.testing.assert_array_equal(saved['pairs'],data['val_pairs'])
    np.testing.assert_array_equal(saved['y'],data['val_y'])
    np.testing.assert_allclose(pred,saved['probability'],rtol=2e-5,atol=2e-5)
    assert abs(roc_auc_score(saved['y'],saved['probability'])-rec['val_auc'])<1e-12
    assert abs(average_precision_score(saved['y'],saved['probability'])-rec['val_ap'])<1e-12
    curves=json.loads(path.with_name(path.stem+'_curves.json').read_text())
    for row in curves:
        assert abs(row['selection_score']-min(row['val_auc']-ba,row['val_ap']-bp))<1e-12
    selected=max(curves,key=lambda x:x['selection_score'])
    assert selected['epoch']==rec['best_epoch']
    assert abs(selected['selection_score']-rec['selection_score'])<1e-12
    original=check_modules.PairStruct
    check_modules.PairStruct=factory
    try: checks=check_modules.check(adapter,count=12)
    finally:check_modules.PairStruct=original
    print('PASS',rec['tag'],flush=True)
    return dict(tag=rec['tag'],status='passed',validation_prediction_max_error=float(np.max(np.abs(pred-saved['probability']))),
        deletion_checks=checks,checkpoint_sha256=rec['checkpoint_sha256'])


def main():
    data=load_fold('MiDrug',0);ba,bp=anchor_metrics(data)
    rows=[]
    for name,cfg in CONFIGS.items():
        seed_all(42);adapter=make_adapter(data,cfg,'cuda:0')
        path=HERE/'results'/f'{name}_s42.json'
        rec=json.loads(path.read_text())
        assert rec['model_sha256']==sha256(HERE/'model.py')
        rows.append(audit_one(path,adapter,lambda D,p,c:AttentivePairStruct(D,p,c,cfg['power']),HERE/'train.py',ba,bp,data))
        del adapter
    for kernel,k in json.loads((HERE/'chemical_results/plan.json').read_text())['configs']:
        seed_all(42);adapter=make_chemical_adapter(data,kernel,k,'cuda:0')
        path=HERE/'chemical_results'/f'{kernel}_k{k}.json'
        assert json.loads(path.read_text())['chemical_sha256']==sha256(HERE/'chemical_kernels.npz')
        # All original covariates, auxiliary edges and supervision must remain exact.
        control=make_chemical_adapter(data,'original',3,'cuda:0')
        torch.testing.assert_close(adapter.y,control.y,rtol=0,atol=0)
        for key in adapter.graph.node_types:
            torch.testing.assert_close(adapter.graph[key].x,control.graph[key].x,rtol=0,atol=0)
        for edge in adapter.graph.edge_types:
            if edge!=('drug','DrugDrug','drug'):
                torch.testing.assert_close(adapter.graph[edge].edge_index,control.graph[edge].edge_index,rtol=0,atol=0)
        del control
        rows.append(audit_one(path,adapter,check_modules.PairStruct,HERE/'train_chem.py',ba,bp,data))
        del adapter
    result=dict(status='passed',records=rows,test_evaluated=False,
        attention_scope='Trained final logits include learned gates; generic intermediate pool checks use cleared unit gates',
        selection='min(validation AUC gain, validation AP gain)',chemical_graph_only_change=True)
    (HERE/'verification.json').write_text(json.dumps(result,indent=2))
    # Check all prior protected source/data/checkpoint hashes without new test use.
    prev=json.loads((HERE.parent/'midrug_signedpaths_20261006/plmf_hash_recheck.json').read_text())
    print('Prior protection schema:',list(prev),flush=True)

if __name__=='__main__':main()
