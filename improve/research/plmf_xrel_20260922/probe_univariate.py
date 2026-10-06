# -*- coding: utf-8 -*-
"""Univariate signal of each XRel column on the seed-42 5cv test folds (fold 0..4): AUC overall, and on rows whose miRNA support degree <= 1 / == 0."""
import os, sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
WARM = '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/plmf_warm_20260922'
from xrel import XRel
ds = sys.argv[1] if len(sys.argv) > 1 else 'MDR'
xr = XRel(ds)
rows = []
for f in range(5):
    z = np.load(os.path.join(WARM, 'cache', f'{ds}_s42_f{f}_ctx+diffK10_kernel10.npz'))
    tr, ytr, te, yte = z['tr'], z['ytr'], z['te'], z['yte']
    sup = tr[ytr == 1]; F = xr.transform(sup, te); dr = np.bincount(sup[:, 0], minlength=xr.nr)[te[:, 0]]
    for c in F.columns:
        v = F[c].values
        r = dict(col=c, auc=roc_auc_score(yte, v))
        for tag, msk in (('deg0', dr == 0), ('deg1', dr <= 1), ('deg2_3', (dr >= 2) & (dr <= 3))):
            r['auc_' + tag] = roc_auc_score(yte[msk], v[msk]) if len(set(yte[msk])) > 1 else np.nan
            r['n_' + tag] = int(msk.sum())
        r['fold'] = f; rows.append(r)
t = pd.DataFrame(rows).groupby('col').mean(numeric_only=True).drop(columns=['fold']).sort_values('auc_deg1', ascending=False)
pd.set_option('display.width', 200); print(ds); print(t.to_string(float_format=lambda x: f'{x:.3f}'))
t.to_csv(os.path.join(HERE, 'results', f'univariate_{ds}.csv'))
