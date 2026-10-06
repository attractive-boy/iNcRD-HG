"""Where does the F + LR(ctx+diff) logit mix gain?  per-positive / per-negative rank-loss change by degree bins."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '../mphgnn_plmf_20260921')
from plmf_splits import get_split, load_dataset
from sklearn.metrics import roc_auc_score as AU
from scipy.special import logit
ds = sys.argv[1]; w = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
mir, dru, pos, neg = load_dataset(ds); nm, nd = len(mir), len(dru)
g = pd.read_csv(f'../plmf_r12_20260927/results/VGHNUNLPPR_{ds}_5cv_seed42_scores.csv')
l = pd.read_csv(f'../plmf_warm_20260922/results/scores_{ds}_s42_ctx+diffK10_lr0.3.csv').rename(columns={'r': 'm'})
M = g.merge(l, on=['fold', 'm', 'd'], suffixes=('_g', '_l'))
M['lg'] = logit(np.clip(M.score_g, 1e-6, 1 - 1e-6)); M['ll'] = logit(np.clip(M.score_l, 1e-6, 1 - 1e-6))
M['mix'] = (1 - w) * M.lg + w * M.ll
out = []
for f in range(5):
    tr, ytr, te, yte = get_split(ds, '5cv', f, 42)
    A = np.zeros((nm, nd)); P = tr[ytr == 1]; A[P[:, 0], P[:, 1]] = 1
    x = M[M.fold == f].copy(); x['dm'] = A.sum(1)[x.m]; x['dd'] = A.sum(0)[x.d]
    for col in ('lg', 'mix'):
        sp = np.sort(x[col][x.y_g == 1].values); sn = np.sort(x[col][x.y_g == 0].values); isp = x.y_g.values == 1
        l_ = np.zeros(len(x)); l_[isp] = 1 - np.searchsorted(sn, x[col].values[isp]) / len(sn); l_[~isp] = np.searchsorted(sp, x[col].values[~isp], side='right') / len(sp)
        x['loss_' + col] = l_
    out.append(x)
X = pd.concat(out)
X['mbin'] = pd.cut(X.dm, [-1, 0, 1, 3, 7, 15, 100]); X['dbin'] = pd.cut(X.dd, [-1, 5, 20, 50, 150, 1000])
for side in ('mbin', 'dbin'):
    for yv in (1, 0):
        x = X[X.y_g == yv]; n = len(x)
        t = x.groupby(side).apply(lambda q: pd.Series(dict(n=len(q) / n, F=q.loss_lg.sum() / n, mix=q.loss_mix.sum() / n)))
        t['delta'] = t.mix - t.F
        print(f'--- {"pos" if yv else "neg"} by {side}'); print(t.round(5).to_string())
