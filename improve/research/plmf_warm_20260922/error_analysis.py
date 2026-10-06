# -*- coding: utf-8 -*-
"""Where does the residual error of ctx+diff x LR live? Bucket test positives/negatives
by training-support degree and by hidden-positive flags; compute per-bucket
contribution to (1-AUC) and the AUC after removing flagged negatives."""
import os, sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921')
from plmf_splits import get_split, load_dataset
HERE = os.path.dirname(os.path.abspath(__file__))

def bucket(x):
    return pd.cut(x, [-1, 0, 1, 3, 7, 15, 10**6], labels=['0', '1', '2-3', '4-7', '8-15', '16+'])

for ds in ('MDR', 'MDS'):
    sc = pd.read_csv(os.path.join(HERE, 'results', f'scores_{ds}_s42_ctx+diff_lr0.3.csv'))
    flags = pd.read_csv(os.path.join(HERE, 'results', f'negflags_{ds}_s42.csv'))
    mir, dru, pos, neg = load_dataset(ds)
    nr, nd = len(mir), len(dru)
    sc['deg_r'] = 0; sc['deg_d'] = 0
    for f in range(5):
        tr, ytr, te, yte = get_split(ds, '5cv', f, 42)
        sup = tr[ytr == 1]
        dr = np.bincount(sup[:, 0], minlength=nr); dd = np.bincount(sup[:, 1], minlength=nd)
        m = sc['fold'] == f
        sc.loc[m, 'deg_r'] = dr[sc.loc[m, 'r'].values]; sc.loc[m, 'deg_d'] = dd[sc.loc[m, 'd'].values]
    sc = sc.merge(flags, on=['r', 'd'], how='left').fillna({'in_other': 0, 'in_midrug': 0})
    print(f'\n================ {ds} ================')
    # per-fold rank-based loss decomposition
    tot_loss = 0; rows = []
    for f in range(5):
        s = sc[sc.fold == f]
        P = s[s.y == 1]; N = s[s.y == 0]
        negs = np.sort(N.score.values)
        # for each positive: fraction of negatives scoring above it
        above = 1 - np.searchsorted(negs, P.score.values, side='left') / len(negs)
        auc = 1 - above.mean()
        poss = np.sort(P.score.values)
        below = np.searchsorted(poss, N.score.values, side='right') / len(poss)  # frac of positives below this negative
        P = P.assign(loss=above); N = N.assign(loss=below)
        rows.append((f, auc, P, N))
    P = pd.concat([r[2] for r in rows]); N = pd.concat([r[3] for r in rows])
    print(f'AUC(mean fold) = {np.mean([r[1] for r in rows]):.4f}; total positive-side loss mass = {P.loss.sum()/len(P):.4f}')
    P['br'] = bucket(P.deg_r); P['bd'] = bucket(P.deg_d)
    N['br'] = bucket(N.deg_r); N['bd'] = bucket(N.deg_d)
    print('\n-- test POSITIVES: share of positives / share of (1-AUC) loss, by miRNA support degree --')
    g = P.groupby('br', observed=True).agg(n=('loss', 'size'), loss=('loss', 'sum'), mean_loss=('loss', 'mean'))
    g['share_n'] = g.n / g.n.sum(); g['share_loss'] = g.loss / g.loss.sum()
    print(g.round(4).to_string())
    print('\n-- by drug support degree --')
    g = P.groupby('bd', observed=True).agg(n=('loss', 'size'), loss=('loss', 'sum'), mean_loss=('loss', 'mean'))
    g['share_n'] = g.n / g.n.sum(); g['share_loss'] = g.loss / g.loss.sum()
    print(g.round(4).to_string())
    print('\n-- joint (miRNA deg x drug deg) mean loss for positives --')
    print(P.pivot_table(index='br', columns='bd', values='loss', aggfunc='mean', observed=True).round(3).to_string())
    print('\n-- joint counts --')
    print(P.pivot_table(index='br', columns='bd', values='loss', aggfunc='size', observed=True).to_string())
    print('\n-- test NEGATIVES: loss by miRNA degree --')
    g = N.groupby('br', observed=True).agg(n=('loss', 'size'), loss=('loss', 'sum'), mean_loss=('loss', 'mean'))
    g['share_n'] = g.n / g.n.sum(); g['share_loss'] = g.loss / g.loss.sum()
    print(g.round(4).to_string())
    print('\n-- negatives flagged as documented positives elsewhere --')
    for col in ('in_other', 'in_midrug'):
        fl = N[N[col] == 1]
        print(f'  {col}: n={len(fl)}, mean loss {fl.loss.mean():.3f} vs unflagged {N[N[col]==0].loss.mean():.3f}; share of neg loss {fl.loss.sum()/N.loss.sum():.3f}')
    # AUC after removing flagged negatives
    for name, mask in (('drop in_other', sc.in_other == 1), ('drop in_other|in_midrug', (sc.in_other == 1) | (sc.in_midrug == 1))):
        keep = sc[~((sc.y == 0) & mask)]
        a = [roc_auc_score(keep[keep.fold == f].y, keep[keep.fold == f].score) for f in range(5)]
        p = [average_precision_score(keep[keep.fold == f].y, keep[keep.fold == f].score) for f in range(5)]
        print(f'  {name}: AUC {np.mean(a):.4f} AUPR {np.mean(p):.4f}')
    # oracle: what if positives with deg_r==0 were perfectly ranked? (upper bound of fixing cold-miRNA-in-warm)
    for bname, bmask in (('deg_r==0', P.deg_r == 0), ('deg_r<=1', P.deg_r <= 1), ('deg_r<=3', P.deg_r <= 3)):
        fixed = P.loss.where(~bmask, 0)
        print(f'  oracle fix positives {bname} (n={bmask.sum()}, {bmask.mean():.3f}): AUC -> {1 - fixed.sum()/len(P):.4f}')
    sc.to_csv(os.path.join(HERE, 'results', f'scores_{ds}_s42_annot.csv'), index=False)
