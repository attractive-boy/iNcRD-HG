"""paired valsel delta of tags vs a reference over available (split, seed, fold)."""
import sys, glob, os, pandas as pd, numpy as np
ref = sys.argv[1]; tags = sys.argv[2:]
for ds in ['MDR', 'MDS']:
    for t in tags:
        rows = []
        for f in glob.glob(f'results/{t}_{ds}_*_seed*.csv'):
            if 'scores' in f or 'curves' in f: continue
            g = f.replace(f'results/{t}_', f'results/{ref}_')
            if not os.path.exists(g): continue
            a, b = pd.read_csv(f), pd.read_csv(g)
            j = a.merge(b, on='fold', suffixes=('', '_r'))
            rows.append(j[['valsel_auc', 'valsel_auc_r', 'valsel_aupr', 'valsel_aupr_r']])
        if not rows: print(ds, t, 'n/a'); continue
        j = pd.concat(rows); da = j.valsel_auc - j.valsel_auc_r; dp = j.valsel_aupr - j.valsel_aupr_r
        print(f'{ds} {t:14s} n={len(j):2d} AUC {j.valsel_auc.mean():.4f} d {da.mean():+.4f} ({(da>0).sum()}/{len(j)}, se {da.std()/np.sqrt(len(j)):.4f}) | AUPR {j.valsel_aupr.mean():.4f} d {dp.mean():+.4f} ({(dp>0).sum()}/{len(j)})')
