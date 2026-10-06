import sys, numpy as np, pandas as pd
sys.path.insert(0, '../mphgnn_plmf_20260921')
from plmf_splits import get_split, load_dataset
from sklearn.metrics import roc_auc_score
ds = sys.argv[1]; tag = sys.argv[2]; rdir = sys.argv[3] if len(sys.argv) > 3 else 'results'
mir, dru, pos, neg = load_dataset(ds); nr, nd = len(mir), len(dru)
sc = pd.read_csv(f'{rdir}/{tag}_{ds}_5cv_seed42_scores.csv'); out = []
for f in range(5):
    tr, ytr, te, yte = get_split(ds, '5cv', f, 42)
    A = np.zeros((nr, nd)); tp = tr[ytr == 1]; A[tp[:, 0], tp[:, 1]] = 1
    s = sc[sc.fold == f].copy(); s['degm'] = A[s.m].sum(1); s['degd'] = A[:, s.d].sum(0); out.append(s)
s = pd.concat(out)
res = []
for d_, g in s.groupby('d'):
    if g.y.nunique() == 2:
        res.append((g.degd.iloc[0], len(g), (g.y == 1).sum(), roc_auc_score(g.y, g.score), roc_auc_score(g.y, g.degm)))
r = pd.DataFrame(res, columns=['degd', 'n', 'npos', 'auc_model', 'auc_degm'])
r['bin'] = pd.cut(r.degd, [-1, 20, 100, 1e9])
print(ds, tag); print(r.groupby('bin').apply(lambda x: pd.Series(dict(ndrugs=len(x), n=x.n.sum(), model=(x.n * x.auc_model).sum() / x.n.sum(), degm=(x.n * x.auc_degm).sum() / x.n.sum()))).round(4))
