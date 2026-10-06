import sys, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '../mphgnn_plmf_20260921'); from plmf_splits import get_split
for ds in ['MDR', 'MDS']:
    sc = pd.read_csv(f'results/F_full_{ds}_5cv_seed42_scores.csv')
    rows = []
    for f in range(5):
        tr, ytr, te, yte = get_split(ds, '5cv', f, 42)
        pos = tr[ytr == 1]
        degm = np.bincount(pos[:, 0], minlength=2000); degd = np.bincount(pos[:, 1], minlength=500)
        s = sc[sc.fold == f].copy(); s['dm'] = degm[s.m.values]; s['dd'] = degd[s.d.values]
        rows.append(s)
    s = pd.concat(rows)
    s['bin'] = pd.cut(s.dm, [-1, 0, 2, 5, 10, 1000])
    print(ds, 'overall', round(roc_auc_score(s.y, s.score), 4), 'pos rate', s.y.mean())
    # per-bin: n, pos rate, AUC within bin, and "loss share": mean logloss share
    s['ll'] = -(s.y * np.log(s.score.clip(1e-6)) + (1 - s.y) * np.log((1 - s.score).clip(1e-6)))
    for b, g in s.groupby('bin'):
        auc = roc_auc_score(g.y, g.score) if g.y.nunique() > 1 else float('nan')
        print(f'  miRNA deg {str(b):10s} n={len(g):5d} pos={g.y.mean():.2f} AUC={auc:.3f} loss share={g.ll.sum() / s.ll.sum():.2f} mean score pos={g.score[g.y==1].mean():.2f} neg={g.score[g.y==0].mean():.2f}')
    s['dbin'] = pd.qcut(s.dd, 4, duplicates='drop')
    for b, g in s.groupby('dbin'):
        print(f'  drug deg {str(b):14s} n={len(g):5d} pos={g.y.mean():.2f} AUC={roc_auc_score(g.y, g.score):.3f} loss share={g.ll.sum() / s.ll.sum():.2f}')
