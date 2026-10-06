"""Error decomposition of a saved score file: AUC loss (mis-ordered pos/neg pairs) by train degree bins."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '../mphgnn_plmf_20260921')
from plmf_splits import get_split, load_dataset
from sklearn.metrics import roc_auc_score
R12 = '../plmf_r12_20260927/results'
tag, ds, sp, seeds = sys.argv[1], sys.argv[2], sys.argv[3], [int(s) for s in sys.argv[4].split(',')]
mir, dru, pos, neg = load_dataset(ds); nm, nd = len(mir), len(dru)
Afull = np.zeros((nm, nd)); Afull[pos[:, 0], pos[:, 1]] = 1
rows = []
for s in seeds:
    sc = pd.read_csv(f'{R12}/{tag}_{ds}_{sp}_seed{s}_scores.csv')
    for f in sorted(sc.fold.unique()):
        tr, ytr, te, yte = get_split(ds, sp, f, s)
        A = np.zeros((nm, nd)); P = tr[ytr == 1]; A[P[:, 0], P[:, 1]] = 1
        x = sc[sc.fold == f].copy()
        x['dm'] = A.sum(1)[x.m.values]; x['dd'] = A.sum(0)[x.d.values]
        x['seed'] = s
        # per-item loss: positive -> frac of negatives scored higher; negative -> frac of positives scored lower
        sp_ = np.sort(x.score[x.y == 1].values); sn_ = np.sort(x.score[x.y == 0].values)
        isp = x.y.values == 1
        l = np.zeros(len(x))
        l[isp] = 1 - np.searchsorted(sn_, x.score.values[isp], side='left') / len(sn_)
        l[~isp] = np.searchsorted(sp_, x.score.values[~isp], side='right') / len(sp_)
        x['loss'] = l
        rows.append(x)
X = pd.concat(rows)
X['mbin'] = pd.cut(X.dm, [-1, 0, 1, 3, 7, 15, 100], labels=['0', '1', '2-3', '4-7', '8-15', '16+'])
X['dbin'] = pd.cut(X.dd, [-1, 5, 20, 50, 150, 1000], labels=['<=5', '6-20', '21-50', '51-150', '151+'])
for side in ('mbin', 'dbin'):
    for yv, nmv in ((1, 'pos'), (0, 'neg')):
        g = X[X.y == yv].groupby(side)
        tot = X[X.y == yv].loss.sum()
        t = pd.DataFrame({'n_frac': g.size() / (X.y == yv).sum(), 'loss_frac': g.loss.sum() / tot, 'mean_loss': g.loss.mean(), 'mean_score': g.score.mean()})
        print(f'--- {nmv} by {side}  (overall 1-AUC = {X[X.y==yv].loss.mean():.4f})'); print(t.round(4).to_string())
# 2D table for positives: loss share
print('--- positive loss share mbin x dbin'); print((X[X.y == 1].pivot_table(index='mbin', columns='dbin', values='loss', aggfunc='sum') / X[X.y == 1].loss.sum()).round(3).to_string())
print('--- positive count share'); print((X[X.y == 1].pivot_table(index='mbin', columns='dbin', values='loss', aggfunc='size') / (X.y == 1).sum()).round(3).to_string())
print('--- negative loss share'); print((X[X.y == 0].pivot_table(index='mbin', columns='dbin', values='loss', aggfunc='sum') / X[X.y == 0].loss.sum()).round(3).to_string())
X.to_pickle(f'analysis/err_{tag}_{ds}_{sp}.pkl')
