"""Replay12 new checkpoints and verify12 immutable reused reference records."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,capture_features,decode
from check import deletion_check,QUERY_CHECK_SOURCE
PREVIOUS=HERE.parent/'midrug_sam_20261006'

def saved_scores(p,data,r):
    with np.load(p.with_name(p.stem+'_inner.npz')) as z:
        for k,dk in [('pairs','val_pairs'),('y','val_y'),('indices','val_indices')]:np.testing.assert_array_equal(z[k],data[dk])
        assert abs(roc_auc_score(z['y'],z['probability'])-r['inner_auc'])<1e-12
        assert abs(average_precision_score(z['y'],z['probability'])-r['inner_ap'])<1e-12
        return z['probability'].copy()

def main():
    plan=json.loads((HERE/'results/plan.json').read_text())
    for p,h in plan['source_hashes'].items():assert sha256(p)==h
    for k,p in [('derivation_sha256',HERE/'README.md'),('initial_checks_sha256',HERE/'initial_checks.json'),
                ('predecessor_review_sha256',HERE/'predecessor_review.json'),('previous_verification_sha256',PREVIOUS/'verification.json'),
                ('previous_selection_sha256',PREVIOUS/'selection.json'),('previous_sensitivity_sha256',PREVIOUS/'sensitivity_diagnostics.json')]:assert sha256(p)==plan[k]
    prior=json.loads((PREVIOUS/'verification.json').read_text());assert prior['status']=='passed'
    for p,h in json.loads((PREVIOUS/'results/plan.json').read_text())['source_hashes'].items():assert sha256(p)==h
    tags={r['tag']:r for r in prior['records']};records=[]
    for fold in range(3):
        data=load_inner(fold)
        for name,sources in plan['controls'].items():
            src=sources[str(fold)];p=Path(src['path']);r=json.loads(p.read_text())
            assert sha256(p)==src['metadata_sha256']
            assert sha256(p.with_suffix('.pt'))==r['checkpoint_sha256']==src['checkpoint_sha256']
            assert sha256(p.with_name(p.stem+'_inner.npz'))==src['probability_sha256']
            assert tags[p.stem]['status']=='passed';saved_scores(p,data,r)
            records.append(dict(tag=p.stem,control_name=name,status='passed',reused_control=True,verification='immutable prior checkpoint/source/probability + current saved metrics/pairs/labels/indices'))
        for name,cfg in CONFIGS.items():
            p=HERE/'results'/f'{name}_f{fold}_s42.json';r=json.loads(p.read_text());assert r['settings']==cfg and r['source_hashes']==plan['source_hashes']
            assert r['graph_sha256']==plan['graph_sha256']==sha256(CLEAN/'graph.pt')
            assert r['data_sha256']==plan['data_sha256']==sha256(CLEAN/'inner_data.npz')
            assert not r['outer_validation_evaluated'] and not r['test_evaluated']
            assert sha256(p.with_suffix('.pt'))==r['checkpoint_sha256']
            seed_all(42);a=make_adapter(data,name,'cuda:0');state=torch.load(p.with_suffix('.pt'),map_location='cuda:0',weights_only=True)
            a.model.load_state_dict(state);a.model.eval()
            assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in k.lower() for k in state)
            assert int(a.model.eb.num_batches_tracked)==r['best_epoch'] and not a.model.lin1._forward_pre_hooks
            with torch.no_grad():
                prob=a.predict('val');logits,h=capture_features(a,'val');decoded=decode(a.model,h)
            torch.testing.assert_close(logits,decoded,atol=2e-5,rtol=2e-5)
            stored=saved_scores(p,data,r);np.testing.assert_allclose(prob,stored,atol=2e-5,rtol=2e-5)
            curves=json.loads(p.with_name(p.stem+'_curves.json').read_text());assert len(curves)==r['epochs_run']
            assert max(curves,key=lambda x:x['selection_score'])['epoch']==r['best_epoch']
            for x in curves:
                assert abs(x['selection_score']-.5*(x['inner_auc']+x['inner_ap']))<1e-12
                assert x['alpha']==cfg['mixture_weight']*min(1.,max(0.,(x['epoch']-30)/30.))
                assert x['reconstruction_weight']==cfg['reconstruction_weight']*min(1.,x['epoch']/50.)
                expected=float(np.float32(1-x['alpha'])*np.float32(x['clean_bce'])+np.float32(x['alpha'])*np.float32(x['perturbed_bce']))
                assert x['association_loss']==expected
                assert x['loss']==float(np.float32(expected)+np.float32(x['reconstruction_weight'])*np.float32(x['reconstruction']))
                assert x['width']==a.model.lin1.in_features and x['total_queries']==len(data['train_y'])
                assert 0<=x['active_queries']<=x['total_queries'] and x['detach_direction']
                assert x['normalized_rms_max_error']<1e-7
                if x['active_queries']:
                    assert abs(x['normalized_rms_mean']-cfg['normalized_rms'])<1e-7
                    if cfg['direction']=='adv':assert abs(x['alignment_mean']-1)<1e-5
                assert np.isfinite(x['scale_min']) and np.isfinite(x['scale_max']) and x['scale_min']>=.001-1e-9
            rec=dict(tag=p.stem,status='passed',reused_control=False,max_prediction_error=float(abs(prob-stored).max()),
                feature_decode_max_logit_error=float(abs(logits-decoded).max()),batchnorm_updates=int(a.model.eb.num_batches_tracked),explicit_deletions=deletion_check(a,data,name))
            bare=make_adapter(data,'control','cuda:0');bare.model.load_state_dict({k:v for k,v in state.items() if not k.startswith('reconstruction.')});bare.model.eval()
            with torch.no_grad():plain=bare.predict('val')
            np.testing.assert_allclose(prob,plain,atol=2e-5,rtol=2e-5);rec['plain_m123_inference_error']=float(abs(prob-plain).max())
            records.append(rec);print('PASS',p.stem,flush=True);del a,bare,state;torch.cuda.empty_cache()
    assert len(records)==24 and sum(not r['reused_control'] for r in records)==12
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=records,new_checkpoint_replays=12,immutable_reused_control_checks=12,
        audit_sources={str(p):sha256(p) for p in [HERE/'audit.py',HERE/'check.py',QUERY_CHECK_SOURCE]},no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
