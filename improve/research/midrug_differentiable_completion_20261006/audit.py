"""Replay six new checkpoints and verify immutable, previously replayed controls."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,load_inner,seed_all,sha256
from model import CONFIGS,PREVIOUS,make_adapter
from check import deletion_check,feature_check,QUERY_CHECK_SOURCE

def saved_scores(path,data,meta):
    with np.load(path.with_name(path.stem+'_inner.npz')) as z:
        for key,dk in [('pairs','val_pairs'),('y','val_y'),('indices','val_indices')]:np.testing.assert_array_equal(z[key],data[dk])
        assert abs(roc_auc_score(z['y'],z['probability'])-meta['inner_auc'])<1e-12
        assert abs(average_precision_score(z['y'],z['probability'])-meta['inner_ap'])<1e-12
        return z['probability'].copy()

def main():
    plan=json.loads((HERE/'results/plan.json').read_text())
    for path,h in plan['source_hashes'].items():assert sha256(path)==h
    for key,path in [('derivation_sha256',HERE/'README.md'),('initial_checks_sha256',HERE/'initial_checks.json'),
                     ('previous_selection_sha256',PREVIOUS/'selection.json'),('previous_verification_sha256',PREVIOUS/'verification.json'),
                     ('predecessor_review_sha256',HERE/'predecessor_review.json')]:assert sha256(path)==plan[key]
    prior=json.loads((PREVIOUS/'verification.json').read_text());assert prior['status']=='passed'
    prior_tags={r['tag']:r for r in prior['records']};records=[]
    for fold in range(3):
        data=load_inner(fold)
        for name,sources in plan['controls'].items():
            src=sources[str(fold)];path=Path(src['path']);r=json.loads(path.read_text())
            assert sha256(path)==src['metadata_sha256']
            assert sha256(path.with_suffix('.pt'))==r['checkpoint_sha256']==src['checkpoint_sha256']
            assert sha256(path.with_name(path.stem+'_inner.npz'))==src['probability_sha256']
            assert prior_tags[path.stem]['status']=='passed';saved_scores(path,data,r)
            records.append(dict(tag=path.stem,status='passed',reused_control=True,verification='immutable prior checkpoint replay plus current saved-score replay',
                previous_replay_max_prediction_error=prior_tags[path.stem]['max_prediction_error']))
        for name,cfg in CONFIGS.items():
            path=HERE/'results'/f'{name}_f{fold}_s42.json';r=json.loads(path.read_text());seed_all(42)
            assert r['settings']==cfg and r['source_hashes']==plan['source_hashes']
            assert not r['outer_validation_evaluated'] and not r['test_evaluated']
            assert sha256(path.with_suffix('.pt'))==r['checkpoint_sha256']
            state=torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True)
            a=make_adapter(data,name,'cuda:0');a.model.load_state_dict(state);a.model.eval()
            assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in k.lower() for k in state)
            with torch.no_grad():prob=a.predict('val')
            stored=saved_scores(path,data,r);np.testing.assert_allclose(prob,stored,atol=2e-5,rtol=2e-5)
            curves=json.loads(path.with_name(path.stem+'_curves.json').read_text())
            assert max(curves,key=lambda x:x['selection_score'])['epoch']==r['best_epoch']
            for x in curves:
                assert abs(x['selection_score']-.5*(x['inner_auc']+x['inner_ap']))<1e-12
                assert x['weight']==cfg['reconstruction_weight']*min(1.,x['epoch']/50.)
                expected=float(np.float32(np.float32(x['bce'])+np.float32(cfg['auxiliary_weight'])*np.float32(x['feature_bce']))+np.float32(x['weight'])*np.float32(x['reconstruction']))
                assert expected==x['loss']
            stop=make_adapter(data,'soft_rec10' if cfg['reconstruction_weight'] else 'soft','cuda:0')
            stop.model.load_state_dict(state);stop.model.eval()
            with torch.no_grad():stopped=stop.predict('val')
            np.testing.assert_allclose(prob,stopped,atol=2e-5,rtol=2e-5)
            rec=dict(tag=path.stem,status='passed',reused_control=False,max_prediction_error=float(abs(prob-stored).max()),
                gradient_switch_forward_error=float(abs(prob-stopped).max()),enumerated_features=feature_check(a,data),
                explicit_deletions=deletion_check(a,data,name))
            if cfg['reconstruction_weight']:
                bare=make_adapter(data,'full','cuda:0');bare.model.load_state_dict({k:v for k,v in state.items() if not k.startswith('reconstruction.')});bare.model.eval()
                with torch.no_grad():out=bare.predict('val')
                np.testing.assert_allclose(prob,out,atol=2e-5,rtol=2e-5);rec['without_decoder_max_error']=float(abs(prob-out).max())
            records.append(rec);print('PASS',path.stem,flush=True)
    assert sum(not r['reused_control'] for r in records)==6 and sum(r['reused_control'] for r in records)==24
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=records,new_checkpoint_replays=6,immutable_reused_control_checks=24,
        audit_sources={str(p):sha256(p) for p in [HERE/'audit.py',HERE/'check.py',PREVIOUS/'check.py',QUERY_CHECK_SOURCE]},
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
