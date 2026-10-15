"""Replay 21 member checkpoints, validate immutable controls, save fixed ensemble."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_inner,seed_all,sha256,make_adapter,member_seed,probability_mean
from check import deletion_check,batch_check,QUERY_CHECK_SOURCE

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
    old_by_path={r['metadata_path']:r for r in prior['records']};records=[];ensembles=[]
    replay=HERE/'replay_predictions';replay.mkdir(exist_ok=True)
    for fold in range(3):
        data=load_inner(fold);members=[];sources=[]
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
            pred=saved_scores(p,data,r)
            records.append(dict(tag=p.stem,metadata_path=str(p),status='passed',reused_control=True,prior_replay_tag=old['tag']))
            if name=='unmasked10':members.append(pred);sources.append(source)
        for member in range(1,8):
            p=HERE/'results'/f'member{member}_f{fold}_r42.json';r=json.loads(p.read_text())
            assert r['config']=='unmasked10' and r['fold']==fold and r['member']==member
            assert r['seed']==member_seed(42,member)==plan['member_seeds'][member]
            assert r['source_hashes']==plan['source_hashes'] and r['plan_sha256']==sha256(HERE/'results/plan.json')
            assert r['graph_sha256']==plan['graph_sha256']==sha256(CLEAN/'graph.pt')
            assert r['data_sha256']==plan['data_sha256']==sha256(CLEAN/'inner_data.npz')
            assert r['no_lncrna'] and not r['outer_validation_evaluated'] and not r['test_evaluated']
            assert sha256(p.with_suffix('.pt'))==r['checkpoint_sha256']
            curves=json.loads(p.with_name(p.stem+'_curves.json').read_text());assert len(curves)==r['epochs_run']
            assert [x['epoch'] for x in curves]==list(range(1,r['epochs_run']+1))
            assert max(curves,key=lambda x:x['selection_score'])['epoch']==r['best_epoch']
            assert curves[r['best_epoch']-1]['selection_score']==r['selection_score']
            assert 300<=r['epochs_run']<=800
            assert r['epochs_run']==800 or r['epochs_run']-r['best_epoch']>=200
            for row in curves:
                assert all(np.isfinite(row[k]) and row[k]>=0 for k in ['inner_auc','inner_ap','bce','reconstruction','reconstruction_rna','reconstruction_drug','loss'])
                assert abs(row['selection_score']-.5*(row['inner_auc']+row['inner_ap']))<1e-12
                assert row['weight']==10*min(1.,row['epoch']/50.)
                assert abs(row['reconstruction']-.5*(row['reconstruction_rna']+row['reconstruction_drug']))<2e-7
                assert row['loss']==float(np.float32(row['bce'])+np.float32(row['weight'])*np.float32(row['reconstruction']))
            seed_all(r['seed']);a=make_adapter(data,'unmasked10','cuda:0')
            state=torch.load(p.with_suffix('.pt'),map_location='cuda:0',weights_only=True);a.model.load_state_dict(state);a.model.eval()
            assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in k.lower() for k in state)
            assert all(torch.isfinite(v).all() for v in state.values())
            assert int(a.model.eb.num_batches_tracked)==r['best_epoch']
            with torch.no_grad():prob=a.predict('val');train=a.predict('train')
            stored=saved_scores(p,data,r);np.testing.assert_allclose(prob,stored,atol=2e-5,rtol=2e-5)
            dest=replay/(p.stem+'.npz');np.savez_compressed(dest,train=train,val=prob)
            bare=make_adapter(data,'control','cuda:0')
            bare.model.load_state_dict({k:v for k,v in state.items() if not k.startswith('reconstruction.')});bare.model.eval()
            with torch.no_grad():plain=bare.predict('val')
            np.testing.assert_allclose(prob,plain,atol=2e-5,rtol=2e-5)
            records.append(dict(tag=p.stem,metadata_path=str(p),status='passed',reused_control=False,
                max_prediction_error=float(abs(prob-stored).max()),batch_max_logit_error=batch_check(a),
                explicit_deletions=deletion_check(a,data,'unmasked10',count=4),
                without_auxiliary_error=float(abs(prob-plain).max()),replay_predictions_sha256=sha256(dest)))
            members.append(stored);sources.append(dict(path=str(p),metadata_sha256=sha256(p),checkpoint_sha256=r['checkpoint_sha256'],probability_sha256=sha256(p.with_name(p.stem+'_inner.npz'))))
            print('PASS',p.stem,flush=True);del a,bare,state;torch.cuda.empty_cache()
        assert len(members)==8 and len(sources)==8
        averaged=probability_mean(members)
        reference=np.array([sum(float(x[i]) for x in members)/8 for i in range(len(averaged))])
        np.testing.assert_array_equal(averaged,reference)
        for ix in [np.arange(7),np.arange(len(averaged)-1,-1,-1)]:
            np.testing.assert_array_equal(probability_mean(np.stack(members)[:,ix]),averaged[ix])
        dest=HERE/'results'/f'ensemble8_f{fold}_r42_inner.npz'
        np.savez_compressed(dest,pairs=data['val_pairs'],y=data['val_y'],indices=data['val_indices'],probability=averaged,member_probabilities=np.stack(members))
        result=dict(fold=fold,config='ensemble8',root_seed=42,members=8,inner_auc=roc_auc_score(data['val_y'],averaged),
            inner_ap=average_precision_score(data['val_y'],averaged),member_sources=sources,probability_sha256=sha256(dest),
            no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
        (HERE/'results'/f'ensemble8_f{fold}_r42.json').write_text(json.dumps(result,indent=2));ensembles.append(result)
    assert len(records)==27 and sum(not r['reused_control'] for r in records)==21
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=records,ensembles=ensembles,
        new_fits=21,new_checkpoint_replays=21,immutable_reused_control_checks=6,
        audit_sources={str(p):sha256(p) for p in [HERE/'audit.py',HERE/'check.py',QUERY_CHECK_SOURCE]},
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
