# -*- coding: utf-8 -*-
"""Paired deltas of every variant vs the PHL reference, per (dataset, split, seed), from results/scores_*.csv."""
import os, re, sys, glob, numpy as np, pandas as pd
REF = sys.argv[1] if len(sys.argv) > 1 else 'phl'
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for f in sorted(glob.glob(os.path.join(HERE, 'results', 'scores_*.csv'))):
    m = re.match(r'scores_(MDR|MDS)_(single)?s(\d+)_K(\d+)_(.+)\.csv', os.path.basename(f))
    if not m: continue
    ds, single, seed, K, var = m.groups(); d = pd.read_csv(f)
    aucs, aps = [], []
    for _, g in d.groupby('fold'):
        aucs.append(roc_auc_score(g['y'], g['score'])); aps.append(average_precision_score(g['y'], g['score']))
    rows.append(dict(ds=ds, split='80/20' if single else '5cv', seed=int(seed), var=var, auc=np.mean(aucs), ap=np.mean(aps)))
t = pd.DataFrame(rows)
ref = t[t['var'] == REF].set_index(['ds', 'split', 'seed'])
out = []
for (ds, split, var), g in t.groupby(['ds', 'split', 'var']):
    da, dp, n = [], [], 0
    for _, r in g.iterrows():
        k = (r['ds'], r['split'], r['seed'])
        if k in ref.index:
            da.append(r['auc'] - ref.loc[k, 'auc']); dp.append(r['ap'] - ref.loc[k, 'ap']); n += 1
    out.append(dict(ds=ds, split=split, var=var, n=len(g), auc=g['auc'].mean(), ap=g['ap'].mean(),
                    dAUC=np.mean(da) if da else np.nan, dAUC_sd=np.std(da) if da else np.nan, dAP=np.mean(dp) if dp else np.nan, dAP_sd=np.std(dp) if dp else np.nan,
                    pos_auc=f'{sum(x > 0 for x in da)}/{n}', pos_ap=f'{sum(x > 0 for x in dp)}/{n}'))
o = pd.DataFrame(out).sort_values(['split', 'ds', 'var'])
pd.set_option('display.width', 200); print(o.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
o.to_csv(os.path.join(HERE, 'results', f'paired_summary_vs_{REF}.csv'), index=False)
