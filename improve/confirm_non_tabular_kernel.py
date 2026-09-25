"""Frozen kernel/GNN composition on three existing, non-selection resplits.

This confirms a post-screen hypothesis; it does not reclassify classical kernel
logistic regression as a frontier invention. Raw two-task outcomes are retained.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.special import logit
import torch

from improve import non_tabular_experiment as base

OUT=base.ROOT/'results'/'non_tabular_kernel_confirmation_20260914'
SPLITS=base.ROOT/'results'/'transfer_splits_20260914'


def prepare(out):
    out=Path(out)
    if out.exists():raise ValueError('Use a new directory')
    parent=base.load_protocol(base.OUT)
    source_names=('confirm_non_tabular_kernel.py','non_tabular_models.py','non_tabular_experiment.py')
    inputs={str(base.OUT/'protocol.json'):base.digest(base.OUT/'protocol.json')}
    for task in base.TASKS:
        inputs[str(base.OUT/'inputs'/task/'arrays.npz')]=base.digest(base.OUT/'inputs'/task/'arrays.npz')
        for split in (42,2026,31415):
            paths=[SPLITS/'inputs'/task/str(split)/'arrays.npz']
            for name in ('gnn','tiered'):
                paths.extend(SPLITS/'runs'/task/str(split)/name/n for n in ('scores.npz','result.json'))
            for path in paths:inputs[str(path)]=base.digest(path)
    p=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
           tasks=list(base.TASKS),splits=[42,2026,31415],seed=20260914,
           fixed_configs={c['id']:c for c in parent['configs'] if c['id'] in ('kernel_1','additive_control')},
           epochs={'MiDrug':{'kernel_1':480,'additive_control':500},
                   'LncDrug':{'kernel_1':210,'additive_control':320}},
           composition='0.5*GNN_logit+0.5*new_model_logit; fixed after historical-inner exploration, before these refits. Both raw standalone and composed outcomes reported.',
           selection='No selection, early stopping, coefficient search or recipe changes on these resplits.',
           transfer_note='Resplits overlap the previously used training pool; they are not independent biological data or untouched external tests.',
           policy='Report a shared two-task modification first. Separately report the explicit MiDrug-only policy retaining the GNN LncDrug branch byte-for-byte; do not hide the raw LncDrug outcomes.',
           gate='MiDrug both mean metrics improve and >=2/3 split wins each; LncDrug mean and every split lose at most 0.001 absolute in each metric. Kernel interaction must also improve over additive-control composition on both mean MiDrug metrics.',
           novelty='Kernel/MAP prior is established literature. This is evidence for a non-tabular engineering extension, not sufficient on its own for the requested three innovative frontier techniques.',
           source_sha256={n:base.digest(base.ROOT/n) for n in source_names},input_sha256=inputs,
           historical_outer_query_labels_access=False,original_test_access=False)
    out.mkdir(parents=True);(out/'source_snapshot').mkdir()
    for n in source_names:shutil.copy2(base.ROOT/n,out/'source_snapshot'/n)
    base.write_json(out/'protocol.json',p)
    print('FROZEN',out,flush=True)


def protocol(out):
    p=json.loads((Path(out)/'protocol.json').read_text())
    for n,h in p['source_sha256'].items():
        if base.digest(base.ROOT/n)!=h:raise ValueError('Source changed: '+n)
    for n,h in p['input_sha256'].items():
        if base.digest(n)!=h:raise ValueError('Input changed: '+n)
    return p


def run(out,task,device):
    out=Path(out);p=protocol(out);torch.set_num_threads(4)
    with np.load(base.OUT/'inputs'/task/'arrays.npz',allow_pickle=False) as z:
        node={k:z[k].copy() for k in ('ur','ud','sr','sd','kr','kd')}
    for split in p['splits']:
        with np.load(SPLITS/'inputs'/task/str(split)/'arrays.npz',allow_pickle=False) as z:
            a={k:z[k].copy() for k in ('pairs_train','y_train','pairs_val','y_val')}
        a.update(node)
        for arm in ('gnn','tiered'):
            with np.load(SPLITS/'runs'/task/str(split)/arm/'scores.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(a['pairs_val'],z['pairs']);np.testing.assert_array_equal(a['y_val'],z['y'])
                a[arm]=z['pred'].copy()
        for name,cfg in p['fixed_configs'].items():
            dest=out/'runs'/task/str(split)/name
            if (dest/'result.json').exists():
                r=json.loads((dest/'result.json').read_text())
                for n,h in r['files'].items():
                    if base.digest(dest/n)!=h:raise ValueError('Result changed')
                continue
            dest.mkdir(parents=True)
            row,pred,state,trace=base.train_one(a,cfg,p['seed'],device,epochs=p['epochs'][task][name],select=False)
            torch.save(state,dest/'model.pt')
            model=base.model_for(a,cfg).to(device)
            model.load_state_dict(torch.load(dest/'model.pt',map_location=device,weights_only=True));model.eval()
            with torch.no_grad():replay=model(torch.as_tensor(a['pairs_val'],device=device)).cpu().numpy()
            np.testing.assert_array_equal(pred,replay)
            scores=dict(standalone=pred,gnn_composed=.5*a['gnn']+.5*pred,
                        tabicl_composed=.5*logit(a['tiered'].astype(np.float64))+.5*pred,
                        gnn=a['gnn'],tabicl=a['tiered'],y=a['y_val'],pairs=a['pairs_val'])
            np.savez_compressed(dest/'scores.npz',**scores)
            base.write_json(dest/'trace.json',trace)
            row.update(task=task,split=split,replay_max_abs=0.,
                       metrics={k:base.metrics(scores['y'],scores[k],scores['pairs'])
                                for k in ('standalone','gnn_composed','tabicl_composed','gnn','tabicl')},
                       files={n:base.digest(dest/n) for n in ('model.pt','scores.npz','trace.json')})
            base.write_json(dest/'result.json',row)
            print('VERIFIED',task,split,name,row['metrics'],flush=True)


def report(out):
    out=Path(out);p=protocol(out);rows=[]
    for task in p['tasks']:
        for split in p['splits']:
            for name in p['fixed_configs']:
                dest=out/'runs'/task/str(split)/name
                r=json.loads((dest/'result.json').read_text())
                for n,h in r['files'].items():
                    if base.digest(dest/n)!=h:raise ValueError('Artifact changed')
                with np.load(dest/'scores.npz',allow_pickle=False) as z:
                    for kind in r['metrics']:
                        m=base.metrics(z['y'],z[kind],z['pairs'])
                        for key,value in m.items():
                            if abs(value-r['metrics'][kind][key])>1e-12:raise ValueError('Metric mismatch')
                        rows.append(dict(task=task,split=split,model=name,kind=kind,**m))
    df=pd.DataFrame(rows);df.to_csv(out/'metrics.csv',index=False)
    contrasts=[]
    for task in p['tasks']:
        for name in p['fixed_configs']:
            for kind,ref in [('standalone','gnn'),('gnn_composed','gnn'),('tabicl_composed','tabicl')]:
                a=df[(df.task==task)&(df.model==name)&(df.kind==kind)].set_index('split')
                b=df[(df.task==task)&(df.model==name)&(df.kind==ref)].set_index('split')
                r=dict(task=task,model=name,kind=kind,reference=ref)
                for metric in ('auc','ap','macro_drug_ap'):
                    delta=a[metric]-b[metric]
                    r.update({metric+'_delta':float(delta.mean()),metric+'_wins':int((delta>0).sum()),
                              metric+'_min_delta':float(delta.min())})
                contrasts.append(r)
    pd.DataFrame(contrasts).to_csv(out/'contrasts.csv',index=False)
    mi=next(x for x in contrasts if x['task']=='MiDrug' and x['model']=='kernel_1' and x['kind']=='gnn_composed')
    lnc=next(x for x in contrasts if x['task']=='LncDrug' and x['model']=='kernel_1' and x['kind']=='gnn_composed')
    control=next(x for x in contrasts if x['task']=='MiDrug' and x['model']=='additive_control' and x['kind']=='gnn_composed')
    mi_gate=all(mi[m+'_delta']>0 and mi[m+'_wins']>=2 for m in ('auc','ap'))
    lnc_gate=all(lnc[m+'_delta']>=-.001 and lnc[m+'_min_delta']>=-.001 for m in ('auc','ap'))
    mechanism=all(mi[m+'_delta']>control[m+'_delta'] for m in ('auc','ap'))
    assessment=dict(mi_performance=mi_gate,lnc_shared_performance=lnc_gate,
                    kernel_interaction_over_additive_control=mechanism,
                    shared_two_task_gate=mi_gate and lnc_gate and mechanism,
                    midrug_only_protected_lnc_gate=mi_gate and mechanism,
                    no_test=True,independent_biological_validation=False,
                    frontier_innovation_count=0,
                    limitation='Classical kernel extension does not by itself satisfy frontier novelty; outcome gates are descriptive.')
    base.write_json(out/'ASSESSMENT.json',assessment)
    base.write_json(out/'COMPLETION.json',dict(runs=12,all_metrics_recomputed=True,all_models_replayed=True,
        artifacts={n:base.digest(out/n) for n in ('metrics.csv','contrasts.csv','ASSESSMENT.json')}))
    print(pd.DataFrame(contrasts).to_string(index=False));print(assessment)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('prepare','run','report'))
    ap.add_argument('--out-dir',default=str(OUT));ap.add_argument('--task',choices=base.TASKS)
    ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    if a.command=='prepare':prepare(a.out_dir)
    elif a.command=='run':run(a.out_dir,a.task,a.device)
    else:report(a.out_dir)


if __name__=='__main__':main()
