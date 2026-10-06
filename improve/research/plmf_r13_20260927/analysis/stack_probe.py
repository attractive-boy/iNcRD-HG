"""Cross-fitted stacking probe on saved test scores: does a candidate pair feature carry information
beyond the model score?  (fit LR on 4 test folds, evaluate on the 5th; diagnostic only)"""
import sys, re, numpy as np, pandas as pd
sys.path.insert(0, '../mphgnn_plmf_20260921')
from plmf_splits import load_dataset, get_split
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
tag, ds = sys.argv[1], sys.argv[2]
X = pd.read_pickle(f'analysis/err_{tag}_{ds}_5cv.pkl')
mir, dru, pos, _ = load_dataset(ds); names = mir.miRNA_Name.astype(str).values
def num(n):
    m = re.search(r'(?:miR|let)-(\d+)', n)
    return int(m.group(1)) if m and 'let' not in n else (7 if 'let' in n else 0)
nums = np.array([num(n) for n in names]); lnum = np.log1p(nums)
X['lg'] = logit(np.clip(X.score, 1e-6, 1 - 1e-6)); X['ln'] = lnum[X.m.values]
X['ldm'] = np.log1p(X.dm); X['ldd'] = np.log1p(X.dd)
# hollow partner-number profile: mean lnum of d's other training miRNAs (exclude m)
X = X.reset_index(drop=True); X['pmu'] = np.nan; X['pz'] = np.nan
for s in sorted(X.seed.unique()):
    for f in range(5):
        tr, ytr, te, yte = get_split(ds, '5cv', f, s)
        P = tr[ytr == 1]; nm, nd = len(names), len(dru)
        A = np.zeros((nm, nd)); A[P[:, 0], P[:, 1]] = 1
        S1 = A.T @ lnum; S2 = A.T @ lnum ** 2; C = A.sum(0)
        sel = (X.seed == s) & (X.fold == f); x = X[sel]
        # test pairs are never in A, so no own-edge exclusion needed
        mu = S1[x.d] / np.maximum(C[x.d], 1); var = S2[x.d] / np.maximum(C[x.d], 1) - mu ** 2
        X.loc[sel, 'pmu'] = mu; X.loc[sel, 'pz'] = (x.ln.values - mu) / np.sqrt(np.maximum(var, 1e-3))
sets = {'score': ['lg'], '+deg': ['lg', 'ldm', 'ldd'], '+num': ['lg', 'ldm', 'ldd', 'ln'],
        '+num+prof': ['lg', 'ldm', 'ldd', 'ln', 'pmu', 'pz']}
for k, cols in sets.items():
    o = []
    for s in sorted(X.seed.unique()):
        x = X[X.seed == s]
        for f in range(5):
            a, b = x[x.fold != f], x[x.fold == f]
            Z = lambda z: np.c_[z[cols].values, z[cols].values * z[['lg']].values]
            c = LogisticRegression(C=10, max_iter=3000).fit(Z(a), a.y); p = c.decision_function(Z(b))
            o.append((AU(b.y, p), AP(b.y, p)))
    print(f'{k:10s} %.4f/%.4f' % tuple(np.mean(o, 0)))
