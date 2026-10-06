"""Diagnostic: is the model miscalibrated ACROSS degree groups?  Cross-fitted per-group Platt scaling on the
saved test scores (fit on the other 4 folds' test scores of the same seed, apply to the held-out fold)."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
tag, ds = sys.argv[1], sys.argv[2]
X = pd.read_pickle(f'analysis/err_{tag}_{ds}_5cv.pkl')
X['lg'] = logit(np.clip(X.score, 1e-6, 1 - 1e-6))
X['ldm'] = np.log1p(X.dm); X['ldd'] = np.log1p(X.dd)
def feats(x, kind):
    L = x.lg.values[:, None]
    if kind == 'platt': return L
    if kind == 'deg': return np.hstack([L, x[['ldm', 'ldd']].values, L * x[['ldm', 'ldd']].values])
    if kind == 'deg2':
        B = np.hstack([x[['ldm', 'ldd']].values, x[['ldm']].values ** 2, x[['ldd']].values ** 2, (x.ldm * x.ldd).values[:, None]])
        return np.hstack([L, B, L * B])
res = {}
for kind in ('none', 'platt', 'deg', 'deg2'):
    out = []
    for s in sorted(X.seed.unique()):
        x = X[X.seed == s]
        for f in range(5):
            trn, tst = x[x.fold != f], x[x.fold == f]
            if kind == 'none':
                p = tst.lg.values
            else:
                clf = LogisticRegression(C=10, max_iter=2000).fit(feats(trn, kind), trn.y)
                p = clf.decision_function(feats(tst, kind))
            out.append((roc_auc_score(tst.y, p), average_precision_score(tst.y, p)))
    res[kind] = np.mean(out, 0)
    print(f'{kind:6s} AUC {res[kind][0]:.4f} AUPR {res[kind][1]:.4f}  d {res[kind][0]-res["none"][0]:+.4f}/{res[kind][1]-res["none"][1]:+.4f}')
