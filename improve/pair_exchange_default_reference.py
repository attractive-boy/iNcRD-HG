"""Post-hoc fixed-default sensitivity control for the pair-exchange experiment.

Requested after the inner-selected pca_full configuration transferred poorly.
No settings are searched here and no checkpoint is selected on outer labels.
The primary experiment and its frozen selection remain untouched.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from improve.analyze_pair_exchange import ranking_metrics
from improve.expression_scale_mechanism import state_digest
from improve.first_round_features import json_digest, sha256, write_json
from improve.pair_exchange import (TASKS, build, checked_result, load_config,
    load_stage, predict, result_dir, train)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',required=True)
    parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args()
    torch.set_num_threads(4)
    out=Path(args.out_dir)
    protocol=load_config(out)
    destination=out/'default_reference'
    destination.mkdir(exist_ok=True)
    specification=dict(parent_protocol_sha256=protocol['sha256'],source_sha256=sha256(__file__),
        purpose='Post-hoc baseline sensitivity after poor transfer of an aggressively tuned pca_full setting; not an additional search or new primary candidate',
        config=dict(id='default_fixed',lr_multiplier=1.,weight_decay=0.),
        epochs={task:protocol['training'][task]['train_epochs'] for task in TASKS},
        seeds=protocol['refit_seeds'],selection='fixed original defaults and full original epoch counts; outer validation only after fitting')
    specification['sha256']=json_digest(specification)
    spec_path=destination/'protocol.json'
    if spec_path.exists():
        if json.loads(spec_path.read_text())!=specification:
            raise ValueError('Default-reference specification changed')
    else:
        write_json(spec_path,specification)
    rows,comparisons,replays=[],[],[]
    for task in TASKS:
        arrays,graph=load_stage(out,protocol,task,'outer')
        for seed in specification['seeds']:
            directory=destination/task/('s'+str(seed))
            directory.mkdir(parents=True,exist_ok=True)
            if not (directory/'result.json').exists():
                print('START fixed_default',task,seed,flush=True)
                result,scores,trace,state=train(arrays,graph,task,'pca_full',seed,specification['config'],
                    'outer',specification['epochs'][task],protocol,args.device)
                result['default_specification_sha256']=specification['sha256']
                np.savez_compressed(directory/'scores.npz',**scores,y_val=arrays['y_val'],pairs_val=arrays['pairs_val'])
                pd.DataFrame(trace).to_csv(directory/'trace.csv',index=False)
                torch.save(state,directory/'checkpoint.pt')
                result['artifacts_sha256']={name:sha256(directory/name) for name in ('scores.npz','trace.csv','checkpoint.pt')}
                write_json(directory/'result.json',result)
            result=checked_result(directory,protocol)
            if result['default_specification_sha256']!=specification['sha256']:
                raise ValueError('Default run used a different specification')
            with np.load(directory/'scores.npz') as saved:
                scores={key:saved[key].copy() for key in saved.files}
            for name in ('y_val','pairs_val'):
                np.testing.assert_array_equal(scores[name],arrays[name])
            metrics=ranking_metrics(scores['y_val'],scores['pairs_val'],scores['val'])
            np.testing.assert_allclose([result['auc'],result['aupr']],[metrics['auc'],metrics['aupr']],rtol=0,atol=1e-12)
            trace=pd.read_csv(directory/'trace.csv')
            np.testing.assert_array_equal(trace.epoch,np.arange(1,specification['epochs'][task]+1))
            if result['selected_epoch']!=specification['epochs'][task] or 'criterion' in trace:
                raise ValueError('Default run must not select an outer checkpoint')
            model,data,pairs,weights=build(arrays,graph,task,'pca_full',seed,protocol,args.device)
            if state_digest(model)!=result['initial_state_sha256']:
                raise ValueError('Default replay initialization differs')
            model.load_state_dict(torch.load(directory/'checkpoint.pt',map_location='cpu'))
            actual=predict(model,data,pairs['val'],weights)['val']
            np.testing.assert_allclose(actual,scores['val'],atol=4e-5,rtol=4e-5)
            replays.append(float(np.max(np.abs(actual-scores['val']))))
            rows.append(dict(task=task,seed=seed,arm='default_fixed',**metrics))
            for arm in ('pca_full','pca_query','factor_exchange'):
                control_dir=result_dir(out,task,arm,'outer','s'+str(seed))
                checked_result(control_dir,protocol)
                with np.load(control_dir/'scores.npz') as saved:
                    candidate=ranking_metrics(saved['y_val'],saved['pairs_val'],saved['val'])
                comparisons.append(dict(task=task,seed=seed,candidate=arm,control='default_fixed',
                    **{metric:candidate[metric]-metrics[metric] for metric in ('auc','aupr','criterion','macro_drug_auc','macro_drug_aupr')}))
            print('VERIFIED fixed_default',task,seed,'%.6f/%.6f'%(metrics['auc'],metrics['aupr']),flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(destination/'results.csv',index=False)
    deltas=pd.DataFrame(comparisons)
    deltas.to_csv(destination/'paired_deltas.csv',index=False)
    means=frame.groupby('task')[['auc','aupr','macro_drug_aupr']].mean()
    delta_means=deltas.groupby(['task','candidate'])[['auc','aupr','macro_drug_aupr']].mean()
    write_json(destination/'VALIDATION.json',dict(passed=True,post_hoc=True,completed_runs=len(rows),
        replayed_runs=len(replays),max_replay_abs=max(replays),test_evaluations=0,
        specification_sha256=specification['sha256']))
    print(means.to_string(),flush=True)
    print(delta_means.to_string(),flush=True)


if __name__=='__main__':
    main()
