"""Recompute every reported metric from predictions and verify all result sources."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from full_adapter import HERE, BENCH, SOURCE, load, sha
from engine import metrics


def audit():
    registry=json.loads((BENCH/'registry.json').read_text())
    original_ids=[x['id'] for x in registry]
    spec=importlib.util.spec_from_file_location('baseline_defaults',BENCH/'run.py')
    defaults=importlib.util.module_from_spec(spec);spec.loader.exec_module(defaults)
    manifest=json.loads((BENCH/'source_manifest.json').read_text())
    for record in manifest['sources']:
        assert sha(record['path'])==record['sha256'],record['path']
        assert sha(BENCH/record['snapshot'])==record['sha256'],record['snapshot']
    dataman=json.loads((BENCH/'data/manifest.json').read_text())
    for p,record in dataman['sources'].items():assert sha(p)==record['sha256'],p
    rows=[];sources=[]
    for model in original_ids+['incrd_m123']:
        folder=HERE/'results' if model=='incrd_m123' else BENCH/'results'
        for ds in ('MiDrug','MDR','MDS'):
            runs=[(0,s) for s in (42,43,44)] if ds=='MiDrug' else [(f,42) for f in range(5)]
            for fold,seed in runs:
                tag=f'{model}_{ds}_f{fold}_s{seed}';path=folder/f'{tag}.json'
                r=json.loads(path.read_text());p=np.load(folder/f'{tag}_predictions.npz')
                assert (r['model'],r['dataset'],r['fold'],r['seed'])==(model,ds,fold,seed)
                assert r['selection']=='validation_auc' and r['graph']=='train_only' and r['threshold']==.5
                config=r['config']
                for k in ('model','dataset','fold','seed'):assert config[k]==r[k]
                if model=='incrd_m123':
                    expected=(300,1e-4,0.,1,300,300)
                    assert config['module_source_sha256']==sha(SOURCE)
                    assert config['modules']['ease_side']=='' and config['modules']['ease']=='10,50'
                else:expected=defaults.DEFAULTS[model]
                for k,v in zip(('epochs','lr','wd','eval_every','min_epochs','patience'),expected):assert config[k]==v,(tag,k)
                assert config['min_epochs']<=r['epochs_run']<=config['epochs']
                data=load(ds,fold)
                for part in ('val','test'):
                    np.testing.assert_array_equal(p[part+'_pairs'],data[part+'_pairs'])
                    np.testing.assert_array_equal(p[part+'_y'],data[part+'_y'])
                    prob=p[part+'_probability']
                    assert prob.shape==data[part+'_y'].shape and np.isfinite(prob).all()
                    assert ((prob>=0)&(prob<=1)).all()
                for k,v in metrics(p['test_y'],p['test_probability']).items():assert abs(r[k]-v)<1e-10,(tag,k)
                curves=json.loads((folder/f'{tag}_curves.json').read_text())
                assert all(not any(k.startswith('test') for k in c) for c in curves)
                best=max(curves,key=lambda c:c['val_auc'])
                assert r['best_epoch']==best['epoch']
                assert abs(metrics(p['val_y'],p['val_probability'])['auc']-r['val_auc'])<1e-10
                assert abs(best['val_auc']-r['val_auc'])<1e-5
                assert sha(folder/f'{tag}.pt')==r['checkpoint_sha256']
                assert sha(BENCH/f'data/{ds}_fold{fold}.npz')==r['split_sha256']
                rows.append({**r,'result_path':str(path),'run_origin':'new_20261004' if model=='incrd_m123' else 'verified_reuse_20260928'})
                artifact_hashes={k:sha(v) for k,v in {
                    'result_json':path,'predictions':folder/f'{tag}_predictions.npz',
                    'curves':folder/f'{tag}_curves.json'}.items()}
                sources.append({**artifact_hashes,'path':str(path),'checkpoint_sha256':r['checkpoint_sha256']})
    assert len(rows)==182
    for model in original_ids+['incrd_m123']:
        for ds in ('MiDrug','MDR','MDS'):
            matched=[r for r in rows if r['model']==model and r['dataset']==ds]
            ref=[r for r in rows if r['model']=='incrd_original' and r['dataset']==ds]
            assert [r['split_sha256'] for r in matched]==[r['split_sha256'] for r in ref]
    result={'verified':True,'reused_runs':169,'new_runs':13,'total_runs':182,
            'models':14,'datasets':3,'complete_combinations':42,'baseline_source_files_verified':len(manifest['sources']),
            'checks':['all saved prediction metrics','sample IDs and labels','split SHA256 identity across all models',
                      'checkpoint SHA256','validation-only best epoch','fixed training configurations',
                      'baseline source snapshots and upstream data hashes'],'artifacts':sources}
    (HERE/'verification.json').write_text(json.dumps(result,indent=2))
    print('PASS: 182/182 runs, 42/42 model-dataset combinations, all metrics recalculated',flush=True)
    return rows

if __name__=='__main__':audit()
