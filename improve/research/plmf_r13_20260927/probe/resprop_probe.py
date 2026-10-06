"""Offline probe 5: residual propagation (Correct-and-Smooth style) from labelled pairs to the query through
shared entities and entity relations.  Cross-fitted LR stacker over the 5 test folds (diagnostic)."""
import sys, re, numpy as np, pandas as pd
sys.path.insert(0, '../mphgnn_plmf_20260921')
from plmf_splits import load_dataset
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'Fd'
g = np.load(f'../mphgnn_plmf_20260921/cache/graph_{ds}.npz', allow_pickle=True)
mir, dru, pos, neg = load_dataset(ds); nm, nd = len(mir), len(dru)
def rowmm(x): mn = x.min(1, keepdims=True); mx = x.max(1, keepdims=True); return (x - mn) / (mx - mn + 1e-8)
def knn(x, k):
    x = rowmm(x); xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12); s = xn @ xn.T; np.fill_diagonal(s, -1e9)
    nb = np.argsort(-s, 1)[:, :k]; W = np.zeros((len(x), len(x))); W[np.repeat(np.arange(len(x)), k), nb.ravel()] = 1; return np.maximum(W, W.T)
def stem(n): return re.sub(r'-(3p|5p)$', '', n.lower().strip())
def family(n):
    s = stem(n).replace('hsa-', ''); s = re.sub(r'-[12]$', '', s); return re.sub(r'([0-9])[a-z]$', r'\1', s)
names = mir.miRNA_Name.astype(str).values
def same(lab): lab = np.asarray(lab); R = (lab[:, None] == lab[None, :]).astype(float); np.fill_diagonal(R, 0); return R
Wm = np.minimum(knn(g['mirna_kmer'], 10) + same([family(n) for n in names]) + same([stem(n) for n in names]), 1); Wd = knn(g['drug_x'], 10)
rows = []
for f in range(5):
    z = np.load(f'cache/dump_{tag}_{ds}_5cv_s42_f{f}.npz')
    P = np.concatenate([z['tr'], z['va']]); Y = np.concatenate([z['ytr'], z['yva']]); L = np.concatenate([z['logit_tr'], z['logit_va']])
    R = np.zeros((nm, nd)); C = np.zeros((nm, nd)); R[P[:, 0], P[:, 1]] = Y - expit(L); C[P[:, 0], P[:, 1]] = 1
    te = z['te']; m, d = te[:, 0], te[:, 1]
    f_ = {}
    f_['r_d'] = R[:, d].sum(0) / (C[:, d].sum(0) + 5); f_['r_m'] = R[m].sum(1) / (C[m].sum(1) + 5)
    WR = Wm @ R; WC = Wm @ C                     # residual of pairs (m', d) with m' related to m
    f_['r_dm'] = WR[m, d] / (WC[m, d] + 3)
    RW = R @ Wd; CW = C @ Wd                     # residual of pairs (m, d') with d' similar to d
    f_['r_md'] = RW[m, d] / (CW[m, d] + 3)
    rows.append(pd.DataFrame(dict(fold=f, y=z['yte'], lg=z['logit_te'], **f_)))
X = pd.concat(rows, ignore_index=True)
def cf(cols):
    o = []
    for f in range(5):
        a, b = X[X.fold != f], X[X.fold == f]
        c = LogisticRegression(C=1.0, max_iter=3000).fit(a[['lg'] + cols].values, a.y); p = c.decision_function(b[['lg'] + cols].values)
        o.append((AU(b.y, p), AP(b.y, p)))
    return np.mean(o, 0)
bs = np.mean([(AU(X[X.fold == f].y, X[X.fold == f].lg), AP(X[X.fold == f].y, X[X.fold == f].lg)) for f in range(5)], 0)
print(f'{ds} base {bs[0]:.4f}/{bs[1]:.4f}')
for cols in (['r_d', 'r_m'], ['r_dm'], ['r_md'], ['r_dm', 'r_md'], ['r_d', 'r_m', 'r_dm', 'r_md']):
    r = cf(cols); print(f'  + {"+".join(cols):22s} {r[0]:.4f}/{r[1]:.4f}  d {r[0]-bs[0]:+.4f}/{r[1]-bs[1]:+.4f}')
