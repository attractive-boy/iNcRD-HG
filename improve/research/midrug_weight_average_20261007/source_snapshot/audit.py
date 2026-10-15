"""Replay six models; verify fixed means, source trajectory and query semantics."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from common import HERE, CLEAN, load_inner, seed_all, sha256
from model import make_adapter, START_EPOCH
from check import deletion_check, QUERY_CHECK_SOURCE


def stored_scores(path, data, meta):
    with np.load(path.with_name(path.stem + '_inner.npz')) as z:
        for key, dk in [('pairs', 'val_pairs'), ('y', 'val_y'), ('indices', 'val_indices')]: np.testing.assert_array_equal(z[key], data[dk])
        p = z['probability'].copy()
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert abs(roc_auc_score(data['val_y'], p) - meta['inner_auc']) < 1e-12
    assert abs(average_precision_score(data['val_y'], p) - meta['inner_ap']) < 1e-12
    return p


@torch.no_grad()
def batch_check(a):
    all_logits = a.logits('val')[:3]
    alone = torch.cat([a.model(a.pairs['val'][:,i:i+1], a.ps, a.qis['val'][i:i+1], a.ease['val'][i:i+1], a.dense) for i in range(3)])
    torch.testing.assert_close(all_logits, alone, atol=2e-4, rtol=5e-5)
    return float(abs(all_logits-alone).max())


@torch.no_grad()
def buffer_check(a, epoch):
    # BN's entire input is fixed EASE, so an isolated reference BN recovers it.
    reference = torch.nn.BatchNorm1d(a.ease['train'].shape[1]).to(a.y.device)
    reference.train()
    for _ in range(epoch): reference(a.ease['train'])
    errors = {}
    for key in ['running_mean', 'running_var', 'num_batches_tracked']:
        v, w = getattr(reference, key), getattr(a.model.eb, key)
        torch.testing.assert_close(v, w, atol=2e-6, rtol=2e-6)
        errors[key] = float(abs(v-w).max())
    return errors


def main():
    plan = json.loads((HERE/'results/plan.json').read_text())
    for path, digest in {**plan['source_hashes'], **plan['evidence_hashes']}.items(): assert sha256(path) == digest
    prior_dir = HERE.parent/'midrug_training_size_20261007'
    prior = json.loads((prior_dir/'verification.json').read_text()); assert prior['status'] == 'passed'
    prior_by_path = {r['metadata_path']:r for r in prior['records']}
    records=[]
    for fold in range(3):
        data=load_inner(fold)
        for name,sources in plan['controls'].items():
            src=sources[str(fold)];p=Path(src['path']);r=json.loads(p.read_text());old=prior_by_path[str(p)]
            assert old['status']=='passed' and old['existing_full_size_model']
            assert sha256(p)==src['metadata_sha256']==old['metadata_sha256']
            assert sha256(p.with_suffix('.pt'))==r['checkpoint_sha256']==src['checkpoint_sha256']==old['checkpoint_sha256']
            assert sha256(p.with_name(p.stem+'_inner.npz'))==src['probability_sha256']
            assert sha256(prior_dir/'replay_predictions'/(old['tag']+'.npz'))==old['replay_predictions_sha256']
            stored_scores(p,data,r)
            records.append(dict(tag=p.stem,metadata_path=str(p),status='passed',reused_control=True,prior_replay_tag=old['tag']))
        raw_curves=json.loads((HERE/'results'/f'raw_f{fold}_s42_curves.json').read_text())
        source=Path(plan['controls']['unmasked10'][str(fold)]['path'])
        source_curves=json.loads(source.with_name(source.stem+'_curves.json').read_text())
        # An independently repeated training trajectory is not a saved-model
        # replay. Record its discrepancy rather than assuming bitwise training
        # reproducibility from a seed (sparse CUDA and gradient allocation differ).
        # The prescribed causal comparison uses raw/mean on THIS same trajectory.
        curve_error={k:max(abs(r[k]-s[k]) for r,s in zip(raw_curves,source_curves)) for k in ['inner_auc','inner_ap','bce','reconstruction']}
        historical_comparison=dict(current_epochs=len(raw_curves),historical_epochs=len(source_curves),
            maximum_errors_over_shared_prefix=curve_error,
            maximum_errors_before_averaging={k:max(abs(r[k]-s[k]) for r,s in zip(raw_curves[:START_EPOCH-1],source_curves[:START_EPOCH-1]))
                                            for k in ['inner_auc','inner_ap','bce','reconstruction']},
            exact_training_reproduction=False,causal_source_of_numerical_difference_not_established=True)
        for name in plan['configs']:
            p=HERE/'results'/f'{name}_f{fold}_s42.json';r=json.loads(p.read_text())
            assert r['config']==name and r['fold']==fold and r['seed']==42
            raw_meta=json.loads((HERE/'results'/f'raw_f{fold}_s42.json').read_text())
            assert r['epochs_run']==raw_meta['epochs_run']
            assert 300<=r['epochs_run']<=800
            assert r['epochs_run']==800 or r['epochs_run']-raw_meta['best_epoch']>=200
            assert r['source_hashes']==plan['source_hashes']
            assert r['graph_sha256']==plan['graph_sha256']==sha256(CLEAN/'graph.pt')
            assert r['data_sha256']==plan['data_sha256']==sha256(CLEAN/'inner_data.npz')
            assert r['no_lncrna'] and not r['outer_validation_evaluated'] and not r['test_evaluated']
            assert sha256(p.with_suffix('.pt'))==r['checkpoint_sha256']
            curves=json.loads(p.with_name(p.stem+'_curves.json').read_text());start=1 if name=='raw' else START_EPOCH
            assert len(curves)==r['epochs_run']-start+1
            assert [x['epoch'] for x in curves]==list(range(start,r['epochs_run']+1))
            assert max(curves,key=lambda x:x['selection_score'])['epoch']==r['best_epoch']
            assert curves[r['best_epoch']-start]['selection_score']==r['selection_score']
            for x in curves:
                assert abs(x['selection_score']-.5*(x['inner_auc']+x['inner_ap']))<1e-12
                assert all(np.isfinite(x[k]) and 0<=x[k]<=1 for k in ['inner_auc','inner_ap'])
                if name=='raw':
                    assert x['weight']==10*min(1.,x['epoch']/50.)
                    assert x['loss']==float(np.float32(x['bce'])+np.float32(x['weight'])*np.float32(x['reconstruction']))
                else:assert x['average_count']==x['epoch']-START_EPOCH+1
            seed_all(42);a=make_adapter(data,name,'cuda:0')
            state=torch.load(p.with_suffix('.pt'),map_location='cuda:0',weights_only=True);a.model.load_state_dict(state);a.model.eval()
            assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in k.lower() for k in state)
            assert int(a.model.eb.num_batches_tracked)==r['best_epoch']
            assert all(torch.isfinite(v).all() for v in state.values())
            mean_error=None
            if name=='mean':
                sums_path=p.with_name(p.stem+'_sums.pt');assert sha256(sums_path)==r['sums_sha256']
                sums=torch.load(sums_path,map_location='cpu',weights_only=True)
                assert sums['selected_count']==r['average_count']==r['best_epoch']-START_EPOCH+1
                assert sums['final_count']==r['epochs_run']-START_EPOCH+1
                params=dict(a.model.named_parameters());assert set(params)==set(sums['selected'])==set(sums['final'])
                for key in params:
                    for part in ['selected','final']:
                        assert sums[part][key].dtype==torch.float64 and torch.isfinite(sums[part][key]).all()
                    expected=(sums['selected'][key]/sums['selected_count']).to(params[key].dtype)
                    torch.testing.assert_close(params[key].cpu(),expected,atol=0,rtol=0)
                mean_error=0.;del sums
                first=curves[0];base=raw_curves[START_EPOCH-1]
                assert first['inner_auc']==base['inner_auc'] and first['inner_ap']==base['inner_ap']
            stored=stored_scores(p,data,r)
            with torch.no_grad():prob=a.predict('val')
            np.testing.assert_allclose(prob,stored,atol=2e-5,rtol=2e-5)
            record=dict(tag=p.stem,metadata_path=str(p),status='passed',reused_control=False,
                max_prediction_error=float(abs(prob-stored).max()),mean_from_saved_sum_error=mean_error,
                historical_trajectory_comparison=historical_comparison if name=='raw' else None,
                buffer_reference_errors=buffer_check(a,r['best_epoch']),explicit_deletions=deletion_check(a,data,name,count=4),batch_max_logit_error=batch_check(a))
            bare=make_adapter(data,'control','cuda:0');bare.model.load_state_dict({k:v for k,v in state.items() if not k.startswith('reconstruction.')});bare.model.eval()
            with torch.no_grad():plain=bare.predict('val')
            np.testing.assert_allclose(prob,plain,atol=2e-5,rtol=2e-5);record['without_reconstruction_error']=float(abs(prob-plain).max())
            records.append(record);print('PASS',p.stem,flush=True);del a,bare,state;torch.cuda.empty_cache()
    assert len(records)==12 and sum(not r['reused_control'] for r in records)==6
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=records,new_training_trajectories=3,new_checkpoint_replays=6,
        immutable_reused_control_checks=6,audit_sources={str(p):sha256(p) for p in [HERE/'audit.py',HERE/'check.py',QUERY_CHECK_SOURCE]},
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))


if __name__=='__main__':main()
