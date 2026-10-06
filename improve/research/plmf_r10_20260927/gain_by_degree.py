"""positive-side ranking loss by miRNA / drug degree (message graph of the reference = 90% training positives), 5cv x 3 seeds"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '../mphgnn_plmf_20260921')
from plmf_splits import get_split, load_dataset
tags = sys.argv[2:]; ds = sys.argv[1]
mir, dru, pos, neg = load_dataset(ds); nr, nd = len(mir), len(dru)
rows = []
for seed in (42, 43, 44):
    S = {t: pd.read_csv(f'results/{t}_{ds}_5cv_seed{seed}_scores.csv') for t in tags}
    for f in range(5):
        tr, ytr, te, yte = get_split(ds, '5cv', f, seed)
        A = np.zeros((nr, nd)); tp = tr[ytr == 1]; A[tp[:, 0], tp[:, 1]] = 1
        base = S[tags[0]][S[tags[0]].fold == f]
        out = pd.DataFrame(dict(y=base.y.values, degm=A[base.m].sum(1), degd=A[:, base.d].sum(0)))
        for t in tags:
            s = S[t][S[t].fold == f].score.values; n = np.sort(s[out.y == 0]); p = np.sort(s[out.y == 1])
            out[t] = np.where(out.y == 1, 1 - np.searchsorted(n, s, 'left') / len(n), np.searchsorted(p, s, 'right') / len(p))
        rows.append(out)
d = pd.concat(rows); P = d[d.y == 1]; N = d[d.y == 0]
for name, X, col, bins in (('pos by miRNA degree', P, 'degm', [-1, 0, 1, 3, 7, 15, 1e9]), ('pos by drug degree', P, 'degd', [-1, 5, 20, 50, 100, 1e9]),
                           ('neg by drug degree', N, 'degd', [-1, 5, 20, 50, 100, 1e9])):
    g = X.groupby(pd.cut(X[col], bins))
    t = g[tags].mean(); t.insert(0, 'share', g.size() / len(X))
    print(ds, name); print(t.round(4).to_string())
