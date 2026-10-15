"""Verify all diagnostic artifacts and independently recompute saved train metrics."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,RELEASE,load_subset,sha256


def main():
    archive=json.loads((HERE/'archive_manifest.json').read_text());plan=json.loads((HERE/'results/plan.json').read_text())
    assessment=json.loads((HERE/'assessment.json').read_text());v=json.loads((HERE/'verification.json').read_text());checks=[]
    for name,digest in archive['source_hashes'].items():
        assert sha256(HERE/name)==digest and sha256(HERE/'source_snapshot'/name)==digest;checks.extend([name,'source_snapshot/'+name])
    for name,digest in archive['artifact_hashes'].items():assert sha256(HERE/name)==digest;checks.append(name)
    for name,digest in archive['frozen_dependencies'].items():assert sha256(name)==digest;checks.append(name)
    assert assessment['plan_sha256']==sha256(HERE/'results/plan.json') and assessment['verification_sha256']==sha256(HERE/'verification.json')
    assert sha256(CLEAN/'inner_data.npz')==plan['data_sha256'] and sha256(CLEAN/'graph.pt')==plan['graph_sha256']
    assert sha256(HERE/'subsets.npz')==plan['subsets_sha256']
    assert v['new_fits']==12 and v['new_checkpoint_replays']==12 and v['existing_checkpoint_replays']==6 and len(v['records'])==18
    result_names=[n for n in archive['artifact_hashes'] if n.startswith('results/') and n!='results/plan.json']
    replay_names=[n for n in archive['artifact_hashes'] if n.startswith('replay_predictions/')]
    assert len(result_names)==48 and len(replay_names)==18
    for row in v['records']:
        metadata=Path(row['metadata_path']);assert sha256(metadata)==row['metadata_sha256']==assessment['point_metadata_hashes'][str(metadata)]
        assert sha256(metadata.with_suffix('.pt'))==row['checkpoint_sha256']
        replay=HERE/'replay_predictions'/f"{row['tag']}.npz";assert sha256(replay)==row['replay_predictions_sha256']
        data=load_subset(row['fold'],row['size'])
        with np.load(replay) as z:
            for name in ['train_pairs','train_y','train_indices','val_pairs','val_y','val_indices']:np.testing.assert_array_equal(z[name],data[name])
            assert abs(roc_auc_score(z['train_y'],z['train_probability'])-row['train_metrics']['auc'])<1e-12
            assert abs(average_precision_score(z['train_y'],z['train_probability'])-row['train_metrics']['ap'])<1e-12
    protection=json.loads((HERE/'plmf_hash_recheck.json').read_text());assert protection['status']=='passed' and len(protection['records'])==10
    refs=HERE.parent/'baselines_m123_20261004/results'
    for r in protection['records']:
        ds,fold=r['dataset'],r['fold']
        assert sha256(refs/f'incrd_m123_{ds}_f{fold}_s42.pt')==r['checkpoint_sha256']
        assert sha256(RELEASE/'data/frozen'/f'{ds}_fold{fold}.npz')==r['split_sha256']
        assert sha256(RELEASE/'data/frozen'/f'{ds}_graph.pt')==r['graph_sha256']
    assert sum(s.startswith('DONE ') for s in (HERE/'launch.log').read_text().splitlines())==12
    assert sum(s.startswith('PASS ') for s in (HERE/'audit.log').read_text().splitlines())==18
    assert (HERE/'check.log').read_text().strip().endswith('ALL PASSED')
    assert assessment['no_candidate_selection'] and assessment['no_performance_extrapolation'] and not assessment['goal_achieved']
    assert not assessment['outer_validation_evaluated'] and not assessment['test_evaluated']
    result=dict(status='passed',hash_checks=len(checks),new_result_artifacts=48,replay_prediction_artifacts=18,new_fits=12,
        new_checkpoint_replays=12,existing_checkpoint_replays=6,protected_plmf_records=10,
        archive_sha256=sha256(HERE/'archive_manifest.json'),assessment_sha256=sha256(HERE/'assessment.json'),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'archive_recheck.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
