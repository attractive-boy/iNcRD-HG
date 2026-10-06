import sys, numpy as np, pandas as pd
sys.path.insert(0, '../mphgnn_plmf_20260921')
from plmf_splits import get_split, load_dataset
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'F_full'
mir, dru, pos, neg = load_dataset(ds); nr, nd = len(mir), len(dru)
sc = pd.read_csv(f'results/{tag}_{ds}_5cv_seed42_scores.csv')
out = []
for f in range(5):
    tr, ytr, te, yte = get_split(ds, '5cv', f, 42)
    A = np.zeros((nr, nd)); tp = tr[ytr == 1]; A[tp[:, 0], tp[:, 1]] = 1
    s = sc[sc.fold == f].copy(); s['degm'] = A[s.m].sum(1); s['degd'] = A[:, s.d].sum(0)
    p = s[s.y == 1].score.values; n = s[s.y == 0].score.values
    ns = np.sort(n); ps_ = np.sort(p)
    s['loss'] = np.where(s.y == 1, 1 - np.searchsorted(ns, s.score, side='left') / len(n), np.searchsorted(ps_, s.score, side='right') / len(p))
    out.append(s)
s = pd.concat(out)
P = s[s.y == 1]; N = s[s.y == 0]
print(ds, tag, 'mean pos loss', P.loss.mean().round(4), 'neg loss', N.loss.mean().round(4))
bm = pd.cut(P.degm, [-1, 0, 1, 3, 7, 15, 1e9]); bd = pd.cut(P.degd, [-1, 5, 20, 50, 100, 1e9])
print(P.groupby(bm).loss.agg(['count', 'mean', 'sum']).assign(share=lambda x: x['sum'] / x['sum'].sum()).round(3))
print(P.groupby(bd).loss.agg(['count', 'mean', 'sum']).assign(share=lambda x: x['sum'] / x['sum'].sum()).round(3))
bm = pd.cut(N.degm, [-1, 0, 1, 3, 7, 15, 1e9]); bd = pd.cut(N.degd, [-1, 5, 20, 50, 100, 1e9])
print('NEG by degm'); print(N.groupby(bm).loss.agg(['count', 'mean', 'sum']).assign(share=lambda x: x['sum'] / x['sum'].sum()).round(3))
print('NEG by degd'); print(N.groupby(bd).loss.agg(['count', 'mean', 'sum']).assign(share=lambda x: x['sum'] / x['sum'].sum()).round(3))
# how much of the loss is 'within-drug' vs 'across-drug'
from sklearn.metrics import roc_auc_score
w = []
for d_, g in s.groupby('d'):
    if g.y.nunique() == 2: w.append((len(g), roc_auc_score(g.y, g.score)))
w = np.array(w); print('within-drug AUC (weighted)', (w[:, 0] * w[:, 1]).sum() / w[:, 0].sum())
w = []
for m_, g in s.groupby('m'):
    if g.y.nunique() == 2: w.append((len(g), roc_auc_score(g.y, g.score)))
w = np.array(w); print('within-miRNA AUC (weighted)', (w[:, 0] * w[:, 1]).sum() / w[:, 0].sum(), len(w))
