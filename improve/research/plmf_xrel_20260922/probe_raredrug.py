# -*- coding: utf-8 -*-
"""Within-rare-drug ranking (seed 42, 5cv): among test rows whose drug has support degree <= tau, AUC of the PHL score,
of each XRel column, and of the ctx+diff columns (by index). Also AUC among rows of a fixed drug (per-drug macro AUC)."""
import os, sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); WARM = '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/plmf_warm_20260922'
from xrel import XRel
ds = sys.argv[1] if len(sys.argv) > 1 else 'MDR'; tau = int(sys.argv[2]) if len(sys.argv) > 2 else 10
xr = XRel(ds); sc = pd.read_csv(os.path.join(HERE, 'results', f'scores_{ds}_s42_K10_phl.csv'))
rows = []
for f in range(5):
    z = np.load(os.path.join(WARM, 'cache', f'{ds}_s42_f{f}_ctx+diffK10_kernel10.npz')); tr, ytr, te, yte, Xte = z['tr'], z['ytr'], z['te'], z['yte'], z['Xte']
    sup = tr[ytr == 1]; dd = np.bincount(sup[:, 1], minlength=xr.nd)[te[:, 1]]; dr = np.bincount(sup[:, 0], minlength=xr.nr)[te[:, 0]]
    s = sc[sc.fold == f]; assert (s.r.values == te[:, 0]).all(); score = s.score.values
    F = xr.transform(sup, te)
    msk = dd <= tau
    def auc(v, m): return roc_auc_score(yte[m], v[m]) if len(set(yte[m])) > 1 else np.nan
    r = dict(fold=f, n=int(msk.sum()), pos=int(yte[msk].sum()), phl=auc(score, msk), phl_all=auc(score, np.ones_like(msk)))
    # per-drug macro AUC (drugs with >=3 pos and >=3 neg in this fold) for rare vs common drugs
    for tag, m2 in (('rare', msk), ('common', ~msk)):
        vals = []
        for d in np.unique(te[m2, 1]):
            mm = m2 & (te[:, 1] == d)
            if yte[mm].sum() >= 3 and (1 - yte[mm]).sum() >= 3: vals.append(roc_auc_score(yte[mm], score[mm]))
        r[f'perdrug_{tag}'] = np.mean(vals) if vals else np.nan; r[f'ndrug_{tag}'] = len(vals)
    for c in F.columns: r['x_' + c] = auc(F[c].values, msk)
    for j in range(Xte.shape[1]): r[f'ctx{j}'] = auc(Xte[:, j], msk)
    rows.append(r)
t = pd.DataFrame(rows).mean(numeric_only=True)
print(f'{ds} rows with support drug degree <= {tau}: n={t.n:.0f}/fold pos={t.pos:.0f}; PHL AUC within-subset {t.phl:.3f} (overall {t.phl_all:.3f}); per-drug macro AUC rare {t.perdrug_rare:.3f} (n={t.ndrug_rare:.0f}) common {t.perdrug_common:.3f} (n={t.ndrug_common:.0f})')
x = t[[i for i in t.index if i.startswith('x_')]].sort_values(ascending=False); print('XRel columns within-subset AUC (top 12):'); print(x.head(12).round(3).to_string())
c = t[[i for i in t.index if i.startswith('ctx')]].sort_values(ascending=False); print('ctx+diff columns within-subset AUC (top 10):'); print(c.head(10).round(3).to_string())
