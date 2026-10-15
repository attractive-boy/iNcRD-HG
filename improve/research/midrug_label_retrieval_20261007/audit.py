"""Replay all retrieval models and test memory exclusion after training."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter
from check import deletion_check,batch_check,memory_check,QUERY_CHECK_SOURCE

def saved_scores(path,data,r):
    with np.load(path.with_name(path.stem+'_inner.npz')) as z:
        for key,dk in [('pairs','val_pairs'),('y','val_y'),('indices','val_indices')]:np.testing.assert_array_equal(z[key],data[dk])
        p=z['probability'].copy()
    assert p.dtype==np.float32 and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    assert abs(roc_auc_score(data['val_y'],p)-r['inner_auc'])<1e-12
    assert abs(average_precision_score(data['val_y'],p)-r['inner_ap'])<1e-12
    return p

def main():
    plan=json.loads((HERE/'results/plan.json').read_text())
    for p,h in {**plan['source_hashes'],**plan['evidence_hashes']}.items():assert sha256(p)==h
    previous=HERE.parent/'midrug_training_size_20261007';prior=json.loads((previous/'verification.json').read_text());assert prior['status']=='passed'
    old_by_path={r['metadata_path']:r for r in prior['records']};records=[]
    replay=HERE/'replay_predictions';replay.mkdir(exist_ok=True)
    for fold in range(3):
        data=load_inner(fold)
        for name,controls in plan['controls'].items():
            source=controls[str(fold)];p=Path(source['path']);r=json.loads(p.read_text());old=old_by_path[str(p)]
            assert old['status']=='passed' and old['existing_full_size_model']
            assert sha256(p)==source['metadata_sha256']==old['metadata_sha256']
            assert sha256(p.with_suffix('.pt'))==r['checkpoint_sha256']==source['checkpoint_sha256']==old['checkpoint_sha256']
            assert sha256(p.with_name(p.stem+'_inner.npz'))==source['probability_sha256']
            assert sha256(previous/'replay_predictions'/(old['tag']+'.npz'))==old['replay_predictions_sha256']
            assert r['seed']==42 and r['fold']==fold and not r['test_evaluated'] and not r['outer_validation_evaluated']
            assert r.get('no_lncrna') is True or r.get('lncrna_used') is False
            assert r['graph_sha256']==plan['graph_sha256'] and r['data_sha256']==plan['data_sha256']
            saved_scores(p,data,r)
            records.append(dict(tag=p.stem,metadata_path=str(p),status='passed',reused_control=True,prior_replay_tag=old['tag']))
        for name in CONFIGS:
            p=HERE/'results'/f'{name}_f{fold}_s42.json';r=json.loads(p.read_text())
            assert r['config']==name and r['fold']==fold and r['seed']==42
            assert r['source_hashes']==plan['source_hashes'] and r['plan_sha256']==sha256(HERE/'results/plan.json')
            assert r['graph_sha256']==plan['graph_sha256']==sha256(CLEAN/'graph.pt')
            assert r['data_sha256']==plan['data_sha256']==sha256(CLEAN/'inner_data.npz')
            assert r['no_lncrna'] and not r['outer_validation_evaluated'] and not r['test_evaluated']
            assert sha256(p.with_suffix('.pt'))==r['checkpoint_sha256']
            curves=json.loads(p.with_name(p.stem+'_curves.json').read_text());assert len(curves)==r['epochs_run']
            assert [x['epoch'] for x in curves]==list(range(1,r['epochs_run']+1))
            assert max(curves,key=lambda x:x['selection_score'])['epoch']==r['best_epoch']
            assert curves[r['best_epoch']-1]['selection_score']==r['selection_score']
            assert 300<=r['epochs_run']<=800 and (r['epochs_run']==800 or r['epochs_run']-r['best_epoch']>=200)
            for row in curves:
                assert all(np.isfinite(row[k]) and row[k]>=0 for k in ['inner_auc','inner_ap','bce','reconstruction','reconstruction_rna','reconstruction_drug','loss'])
                assert abs(row['selection_score']-.5*(row['inner_auc']+row['inner_ap']))<1e-12
                assert row['weight']==10*min(1.,row['epoch']/50.)
                assert abs(row['reconstruction']-.5*(row['reconstruction_rna']+row['reconstruction_drug']))<2e-7
                assert row['loss']==float(np.float32(row['bce'])+np.float32(row['weight'])*np.float32(row['reconstruction']))
            seed_all(42);a=make_adapter(data,name,'cuda:0')
            state=torch.load(p.with_suffix('.pt'),map_location='cuda:0',weights_only=True);a.model.load_state_dict(state);a.model.eval()
            assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in k.lower() for k in state)
            assert all(torch.isfinite(v).all() for v in state.values())
            assert int(a.model.eb.num_batches_tracked)==r['best_epoch']
            with torch.no_grad():prob=a.predict('val');train=a.predict('train')
            stored=saved_scores(p,data,r);np.testing.assert_allclose(prob,stored,atol=2e-5,rtol=2e-5)
            dest=replay/(p.stem+'.npz');np.savez_compressed(dest,train=train,val=prob)
            flipped=make_adapter({**data,'val_y':1-data['val_y']},name,'cuda:0');flipped.model.load_state_dict(state);flipped.model.eval()
            with torch.no_grad():alternative=flipped.predict('val')
            np.testing.assert_allclose(prob,alternative,atol=2e-5,rtol=2e-5)
            records.append(dict(tag=p.stem,metadata_path=str(p),status='passed',reused_control=False,
                max_prediction_error=float(abs(prob-stored).max()),batch_max_logit_error=batch_check(a),memory_check=memory_check(a),
                explicit_deletions=deletion_check(a,data,name,count=4),heldout_label_flip_error=float(abs(prob-alternative).max()),
                replay_predictions_sha256=sha256(dest)))
            print('PASS',p.stem,flush=True);del a,flipped,state;torch.cuda.empty_cache()
    assert len(records)==15 and sum(not r['reused_control'] for r in records)==9
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=records,new_fits=9,new_checkpoint_replays=9,immutable_reused_control_checks=6,
        audit_sources={str(p):sha256(p) for p in [HERE/'audit.py',HERE/'check.py',QUERY_CHECK_SOURCE]},
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
