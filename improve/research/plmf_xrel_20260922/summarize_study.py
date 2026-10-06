# -*- coding: utf-8 -*-
"""Aggregate study-grouped split lines (results/study_split.txt) into a table: mean over seeds per (protocol, dataset, variant), paired deltas vs a reference."""
import re, sys, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REF = sys.argv[1] if len(sys.argv) > 1 else 'phl'
rows = []
for l in open(os.path.join(HERE, 'results', 'study_split.txt')):
    m = re.match(r'## (STUDY-GROUPED\w*) (MDR|MDS) s(\d+) K10 (\S+) ncol=\d+: AUC ([\d.]+)±[\d.]+ AUPR ([\d.]+)', l)
    if m: rows.append(dict(proto=m.group(1), ds=m.group(2), seed=int(m.group(3)), var=m.group(4), auc=float(m.group(5)), ap=float(m.group(6))))
t = pd.DataFrame(rows).drop_duplicates(['proto', 'ds', 'seed', 'var'], keep='last')
ref = t[t['var'] == REF].set_index(['proto', 'ds', 'seed'])
out = []
for (p, ds, v), g in t.groupby(['proto', 'ds', 'var']):
    da, dp = [], []
    for _, r in g.iterrows():
        k = (p, ds, r['seed'])
        if k in ref.index: da.append(r['auc'] - ref.loc[k, 'auc']); dp.append(r['ap'] - ref.loc[k, 'ap'])
    out.append(dict(proto=p, ds=ds, var=v, n=len(g), auc=g['auc'].mean(), ap=g['ap'].mean(), dAUC=np.mean(da) if da else np.nan, dAP=np.mean(dp) if dp else np.nan, pos=f'{sum(x > 0 for x in da)}/{len(da)}'))
o = pd.DataFrame(out).sort_values(['proto', 'ds', 'var']); pd.set_option('display.width', 200)
print(o.to_string(index=False, float_format=lambda x: f'{x:.4f}')); o.to_csv(os.path.join(HERE, 'results', f'study_summary_vs_{REF}.csv'), index=False)
