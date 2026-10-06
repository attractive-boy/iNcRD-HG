"""LR on the GNN's own hollow pair statistics (qis + ease), trained directly on the training pairs (no cross-fitting),
and its rank-average with the GNN."""
import sys, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
ds, tag = sys.argv[1], sys.argv[2]
sc = pd.read_csv(f'results/{tag}_{ds}_5cv_seed42_scores.csv')
res = {k: [] for k in ('lr', 'gnn', 'avg', 'lr_ctx', 'avg_ctx')}
for f in range(5):
    z = np.load(f'cache/feats_{ds}_5cv_s42_f{f}.npz')
    ease_sd = z['ease_tr'].std(0) + 1e-6; ease_mu = z['ease_tr'].mean(0)
    Xtr = np.column_stack([z['qis_tr'], (z['ease_tr'] - ease_mu) / ease_sd]); Xte = np.column_stack([z['qis_te'], (z['ease_te'] - ease_mu) / ease_sd])
    clf = LogisticRegression(C=0.3, max_iter=5000).fit(Xtr, z['ytr']); s = clf.decision_function(Xte)
    g = sc[sc.fold == f]; assert (g.m.values == z['te'][:, 0]).all(); y = z['yte']
    zz = np.load(f'../plmf_warm_20260922/cache/{ds}_s42_f{f}_ctx+diffK10_kernel10.npz')
    s2 = LogisticRegression(C=0.3, max_iter=5000).fit(zz['Xtr'], zz['ytr']).decision_function(zz['Xte'])
    for k, v in (('lr', s), ('gnn', g.score.values), ('avg', rankdata(s) + rankdata(g.score.values)), ('lr_ctx', s2), ('avg_ctx', rankdata(s2) + rankdata(g.score.values))):
        res[k].append((AU(y, v), AP(y, v)))
print(ds, tag, {k: tuple(np.round(np.mean(v, 0), 4)) for k, v in res.items()})
