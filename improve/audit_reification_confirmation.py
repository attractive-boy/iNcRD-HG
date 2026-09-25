"""Shared endpoint perturbations for the frozen graph-transfer confirmation.

These intervals condition on trained models and reused development partitions.
They are neither external validation nor post-selection significance tests.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score

from improve import confirm_reification_kernel as experiment
from improve import non_tabular_experiment as base
from improve import reification_transfer as graph


def weighted_metrics(y,pred,w):
    return np.array([roc_auc_score(y,pred,sample_weight=w),
                     average_precision_score(y,pred,sample_weight=w)])


def audit(out,repetitions=1000):
    out=Path(out);p=experiment.protocol(out)
    complete=json.loads((out/'COMPLETION.json').read_text())
    for n,h in complete['files'].items():
        if base.digest(out/n)!=h:raise ValueError('Confirmation report changed')
    contrasts={'gnn_composed_vs_gnn':('pretrained','gnn'),
               'gnn_composed_vs_kernel':('pretrained','kernel'),
               'gnn_composed_vs_random':('pretrained','random'),
               'standalone_vs_gnn':('standalone','gnn'),
               'tabicl_composed_vs_tabicl':('tabicl_composed','tabicl')}
    inputs={str(out/'COMPLETION.json'):base.digest(out/'COMPLETION.json')}
    results={};overlap={};draw_arrays={}
    for task in p['tasks']:
        data=[]
        nr=nd=None
        for split in p['splits']:
            a=experiment.load_inputs(out,task,split)
            nr,nd=int(a['nr']),int(a['nd'])
            item=dict(split=split,y=a['y_val'],pairs=a['pairs_val'])
            for kind in p['kinds']:
                dest=out/'runs'/task/str(split)/kind
                r=json.loads((dest/'result.json').read_text());graph.verify_files(dest,r)
                for name in ('result.json','scores.npz'):inputs[str(dest/name)]=base.digest(dest/name)
                with np.load(dest/'scores.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(item['y'],z['y'])
                    np.testing.assert_array_equal(item['pairs'],z['pairs'])
                    item[kind]=z['gnn_composed'].copy()
                    if kind=='pretrained':
                        for n in ('gnn','tabicl','standalone','tabicl_composed'):item[n]=z[n].copy()
            data.append(item)
        overlap[task]=[]
        for left,right in combinations(data,2):
            l=set(map(tuple,left['pairs'].T));r=set(map(tuple,right['pairs'].T))
            overlap[task].append(dict(splits=[left['split'],right['split']],
                                      shared_pairs=len(l&r),left_pairs=len(l),right_pairs=len(r),
                                      shared_rnas=len(set(left['pairs'][0])&set(right['pairs'][0])),
                                      shared_drugs=len(set(left['pairs'][1])&set(right['pairs'][1]))))
        rng=np.random.default_rng(202609142+(task=='MiDrug'))
        names=sorted(set(v for pair in contrasts.values() for v in pair))
        draws={n:[] for n in contrasts}
        for _ in range(repetitions):
            rw=rng.poisson(1,nr);dw=rng.poisson(1,nd)
            split_metrics=[]
            for item in data:
                w=rw[item['pairs'][0]]*dw[item['pairs'][1]]
                if any(w[item['y']==c].sum()<=0 for c in (0,1)):break
                split_metrics.append({n:weighted_metrics(item['y'],item[n],w) for n in names})
            if len(split_metrics)!=len(data):continue
            for contrast,(new,ref) in contrasts.items():
                draws[contrast].append(np.mean([row[new]-row[ref] for row in split_metrics],axis=0))
        results[task]={}
        for contrast,(new,ref) in contrasts.items():
            values=np.stack(draws[contrast]);draw_arrays[task+'__'+contrast]=values
            point=np.mean([weighted_metrics(item['y'],item[new],np.ones(len(item['y'])))-
                           weighted_metrics(item['y'],item[ref],np.ones(len(item['y']))) for item in data],axis=0)
            results[task][contrast]={metric:dict(point=float(point[i]),
                quantile_025=float(np.quantile(values[:,i],.025)),quantile_975=float(np.quantile(values[:,i],.975)),
                resamples=len(values)) for i,metric in enumerate(('auc','ap'))}
        print('AUDITED',task,flush=True)
    dest=out/'uncertainty';dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'draws.npz',**draw_arrays)
    report=dict(method='RNA and drug Poisson(1) weights multiplied at each pair. The same RNA/drug weights are used for every overlapping split and every compared predictor within one replicate.',
        requested_repetitions=repetitions,model_refitting=False,source_sha256=base.digest(__file__),
        original_test_access=False,independent_biological_validation=False,
        interpretation='Descriptive endpoint perturbation intervals conditional on fixed models; no adjustment for extensive earlier model search and no formal noninferiority or significance claim.',
        input_sha256=inputs,split_overlap=overlap,results=results,
        draws_sha256=base.digest(dest/'draws.npz'))
    base.write_json(dest/'UNCERTAINTY.json',report)
    print(json.dumps(results,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir',default=str(experiment.OUT));ap.add_argument('--repetitions',type=int,default=1000)
    args=ap.parse_args()
    if args.repetitions<100:ap.error('Use at least 100 perturbations')
    audit(args.out_dir,args.repetitions)


if __name__=='__main__':main()
