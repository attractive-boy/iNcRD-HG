"""usage: python summ.py REF TAG [TAG ...]  -- valsel AUC/AUPR per dataset/split/seed, paired delta vs REF"""
import sys, glob, os, numpy as np, pandas as pd
ref = sys.argv[1]; tags = sys.argv[2:]
R = 'results'
rows = []
for f in sorted(glob.glob(f'{R}/*.csv')):
    b = os.path.basename(f)
    if b.endswith('_scores.csv'): continue
    d = pd.read_csv(f); rows.append(d)
D = pd.concat(rows)
for (ds, sp, sd), g in D.groupby(['dataset', 'split', 'seed']):
    r = g[g.tag == ref].set_index('fold')
    if r.empty: continue
    print(f'{ds} {sp} seed{sd}: {ref} {r.valsel_auc.mean():.4f}/{r.valsel_aupr.mean():.4f}')
    for t in tags:
        x = g[g.tag == t].set_index('fold')
        if x.empty: continue
        i = r.index.intersection(x.index)
        da = x.valsel_auc[i] - r.valsel_auc[i]; dp = x.valsel_aupr[i] - r.valsel_aupr[i]
        print(f'   {t:28s} {x.valsel_auc[i].mean():.4f}/{x.valsel_aupr[i].mean():.4f}  d {da.mean():+.4f}/{dp.mean():+.4f} ({(da>0).sum()}/{len(i)}·{(dp>0).sum()}/{len(i)})  best {x.best_auc[i].mean():.4f} last {x.last_auc[i].mean():.4f}')
