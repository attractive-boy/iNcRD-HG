import sys, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
ds = sys.argv[1]
D = np.arange(33, 85).reshape(13, 4)
sets = {'qe': None, 'ctx': np.arange(33), 'diff': D.ravel(), 'ctxdiff': np.arange(85), 'qe+ctx': np.arange(33), 'qe+diff': D.ravel(), 'qe+ctxdiff': np.arange(85),
        'qe+diffraw': D[:, 0], 'qe+diffrz': D[:, [1, 2, 3]].ravel(), 'qe+ctx_nocontent': np.r_[0:25, 31:33], 'qe+svd': np.array([12])}
res = {k: [] for k in sets}
for f in range(5):
    z = np.load(f'cache/feats_{ds}_5cv_s42_f{f}.npz'); zz = np.load(f'../plmf_warm_20260922/cache/{ds}_s42_f{f}_ctx+diffK10_kernel10.npz')
    key = lambda P: P[:, 0] * 1000 + P[:, 1]
    pos = pd.Series(np.arange(len(zz['tr'])), index=key(zz['tr']))
    itr = pos[key(z['tr'])].values
    assert (zz['te'] == z['te']).all()
    em, es = z['ease_tr'].mean(0), z['ease_tr'].std(0) + 1e-6
    QEtr = np.column_stack([z['qis_tr'], (z['ease_tr'] - em) / es]); QEte = np.column_stack([z['qis_te'], (z['ease_te'] - em) / es])
    for k, idx in sets.items():
        if k == 'qe': Xtr, Xte = QEtr, QEte
        elif k.startswith('qe+'): Xtr, Xte = np.column_stack([QEtr, zz['Xtr'][itr][:, idx]]), np.column_stack([QEte, zz['Xte'][:, idx]])
        else: Xtr, Xte = zz['Xtr'][itr][:, idx], zz['Xte'][:, idx]
        s = LogisticRegression(C=0.3, max_iter=6000).fit(Xtr, z['ytr']).decision_function(Xte)
        res[k].append((AU(z['yte'], s), AP(z['yte'], s)))
for k, v in res.items(): print(ds, f'{k:18s}', tuple(np.round(np.mean(v, 0), 4)))
