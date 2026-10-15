"""Replay nine masked-context teacher models and preserve immutable controls."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,load_masks
from check import deletion_check,QUERY_CHECK_SOURCE,context_check


def saved_scores(path,data,r):
    with np.load(path.with_name(path.stem+'_inner.npz')) as z:
        for key,dk in [('pairs','val_pairs'),('y','val_y'),('indices','val_indices')]:np.testing.assert_array_equal(z[key],data[dk])
        p=z['probability'].copy()
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    assert abs(roc_auc_score(data['val_y'],p)-r['inner_auc'])<1e-12
    assert abs(average_precision_score(data['val_y'],p)-r['inner_ap'])<1e-12
    return p


@torch.no_grad()
def batch_check(a):
    batched=a.logits('val')[:3]
    alone=torch.cat([a.model(a.pairs['val'][:,i:i+1],a.ps,a.qis['val'][i:i+1],a.ease['val'][i:i+1],a.dense) for i in range(3)])
    torch.testing.assert_close(batched,alone,atol=2e-4,rtol=5e-5)
    return float(abs(batched-alone).max())




def main():
    plan=json.loads((HERE/'results/plan.json').read_text())
    for p,h in {**plan['source_hashes'],**plan['evidence_hashes']}.items():assert sha256(p)==h
    previous=HERE.parent/'midrug_training_size_20261007';prior=json.loads((previous/'verification.json').read_text());assert prior['status']=='passed'
    old_by_path={r['metadata_path']:r for r in prior['records']};records=[]
    for fold in range(3):
        data=load_inner(fold)
        for name,sources in plan['controls'].items():
            source=sources[str(fold)];p=Path(source['path']);r=json.loads(p.read_text());old=old_by_path[str(p)]
            assert old['status']=='passed' and old['existing_full_size_model']
            assert sha256(p)==source['metadata_sha256']==old['metadata_sha256']
            assert sha256(p.with_suffix('.pt'))==r['checkpoint_sha256']==source['checkpoint_sha256']==old['checkpoint_sha256']
            assert sha256(p.with_name(p.stem+'_inner.npz'))==source['probability_sha256']
            assert sha256(previous/'replay_predictions'/(old['tag']+'.npz'))==old['replay_predictions_sha256']
            saved_scores(p,data,r)
            records.append(dict(tag=p.stem,metadata_path=str(p),status='passed',reused_control=True,prior_replay_tag=old['tag']))
        for name in CONFIGS:
            p=HERE/'results'/f'{name}_f{fold}_s42.json';r=json.loads(p.read_text())
            assert r['masks_sha256']==plan['masks_sha256']==sha256(HERE/'masks.npz')
            assert r['config']==name and r['fold']==fold and r['seed']==42
            assert r['source_hashes']==plan['source_hashes']
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
                assert all(np.isfinite(row[k]) and row[k]>=0 for k in ['inner_auc','inner_ap','bce','reconstruction','latent','latent_miRNA','latent_drug','loss'])
                assert abs(row['selection_score']-.5*(row['inner_auc']+row['inner_ap']))<1e-12
                assert row['reconstruction_weight']==10*min(1.,row['epoch']/50.)
                expected=np.float32(row['bce'])+np.float32(row['reconstruction_weight'])*np.float32(row['reconstruction'])
                assert row['latent_weight']==.1*min(1.,max(0.,(row['epoch']-30)/30.))
                assert row['teacher_updates']==row['epoch']
                assert abs(row['latent']-.5*(row['latent_miRNA']+row['latent_drug']))<2e-7
                if row['latent_weight']>0:expected=expected+np.float32(row['latent_weight'])*np.float32(row['latent'])
                assert row['loss']==float(expected)
            seed_all(42);a=make_adapter(data,name,'cuda:0')
            state=torch.load(p.with_suffix('.pt'),map_location='cuda:0',weights_only=True);a.model.load_state_dict(state);a.model.eval()
            assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in k.lower() for k in state)
            assert all(torch.isfinite(v).all() for v in state.values())
            assert int(a.model.eb.num_batches_tracked)==r['best_epoch']
            assert int(a.model.context_teacher.teacher_updates)==r['best_epoch']
            student=dict(a.model.encoder.original.named_parameters())
            for key,value in a.model.context_teacher.teacher.named_parameters():
                assert not value.requires_grad and value.grad is None
                if name=='block_current':torch.testing.assert_close(value,student[key],atol=0,rtol=0)
            schedule=load_masks(fold,'cuda:0')[CONFIGS[name]['mask']]
            masked={n:x[r['best_epoch']-1] for n,x in schedule.items()}
            with torch.no_grad():prob=a.predict('val')
            stored=saved_scores(p,data,r);np.testing.assert_allclose(prob,stored,atol=2e-5,rtol=2e-5)
            item=dict(tag=p.stem,metadata_path=str(p),status='passed',reused_control=False,
                max_prediction_error=float(abs(prob-stored).max()),context_check=context_check(a,masked),
                explicit_deletions=deletion_check(a,data,name,count=4),batch_max_logit_error=batch_check(a))
            bare=make_adapter(data,'control','cuda:0')
            bare.model.load_state_dict({k:v for k,v in state.items() if not k.startswith(('reconstruction.','context_teacher.'))});bare.model.eval()
            with torch.no_grad():plain=bare.predict('val')
            np.testing.assert_allclose(prob,plain,atol=2e-5,rtol=2e-5);item['without_auxiliary_modules_error']=float(abs(prob-plain).max())
            records.append(item);print('PASS',p.stem,flush=True);del a,bare,state;torch.cuda.empty_cache()
    assert len(records)==15 and sum(not r['reused_control'] for r in records)==9
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=records,new_fits=9,new_checkpoint_replays=9,
        immutable_reused_control_checks=6,audit_sources={str(p):sha256(p) for p in [HERE/'audit.py',HERE/'check.py',QUERY_CHECK_SOURCE]},
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))


if __name__=='__main__':main()
