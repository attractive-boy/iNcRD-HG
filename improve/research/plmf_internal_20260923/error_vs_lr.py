# -*- coding: utf-8 -*-
"""Where does the GNN lose to the internal LR (ctx+diff x LR)?  Per-fold ranking loss of every test
pair (positives: share of negatives scored above it; negatives: share of positives scored below it),
bucketed by the miRNA / drug training degree.  usage: python error_vs_lr.py TAG [TAG2 ...]"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, 'mphgnn_plmf_20260921'))
from plmf_splits import get_split  # noqa

tags = sys.argv[1:] or ['pair']
BK = [(0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 10 ** 6)]
DK = [(0, 5), (6, 30), (31, 100), (101, 10 ** 6)]


def rank_loss(y, s):
    pos, neg = s[y == 1], s[y == 0]
    ns = np.sort(neg); ps = np.sort(pos)
    lp = 1 - np.searchsorted(ns, pos, side='left') / len(neg) - 0.5 * (np.searchsorted(ns, pos, 'right') - np.searchsorted(ns, pos, 'left')) / len(neg)
    ln = np.searchsorted(ps, neg, side='right') / len(pos) - 0.5 * (np.searchsorted(ps, neg, 'right') - np.searchsorted(ps, neg, 'left')) / len(pos)
    out = np.zeros(len(y)); out[y == 1] = lp; out[y == 0] = ln
    return out


for ds in ('MDR', 'MDS'):
    frames = {}
    lr = pd.read_csv(os.path.join(BASE, 'plmf_warm_20260922', 'results', f'scores_{ds}_s42_ctx+diff_lr0.3.csv')).rename(columns={'r': 'm'})
    frames['LR'] = lr
    for t in tags:
        frames[t] = pd.read_csv(os.path.join(HERE, 'results', f'{t}_{ds}_5cv_seed42_scores.csv'))
    rows = []
    for fold in range(5):
        tr, ytr, te, yte = get_split(ds, '5cv', fold, 42)
        pos = tr[ytr == 1]
        degm = np.bincount(pos[:, 0], minlength=2000); degd = np.bincount(pos[:, 1], minlength=500)
        base = pd.DataFrame({'m': te[:, 0], 'd': te[:, 1], 'y': yte.astype(int)})
        for name, f in frames.items():
            s = f[f.fold == fold].merge(base, on=['m', 'd', 'y'], how='right')['score'].values
            L = rank_loss(base.y.values, s)
            base[name] = L
        base['degm'] = degm[base.m]; base['degd'] = degd[base.d]; base['fold'] = fold
        rows.append(base)
    df = pd.concat(rows)
    print(f'== {ds}  mean ranking loss (=1-AUC) by miRNA train degree; share of test positives/negatives')
    for side, yv in (('pos', 1), ('neg', 0)):
        sub = df[df.y == yv]
        for lo, hi in BK:
            b = sub[(sub.degm >= lo) & (sub.degm <= hi)]
            print(f'  {side} degm {lo:>2}-{hi if hi < 1e5 else "+":<3} share {len(b) / len(sub):.2f} ' + ' '.join(f'{k} {b[k].mean():.3f}' for k in frames))
    print(f'  -- by drug train degree')
    for side, yv in (('pos', 1), ('neg', 0)):
        sub = df[df.y == yv]
        for lo, hi in DK:
            b = sub[(sub.degd >= lo) & (sub.degd <= hi)]
            print(f'  {side} degd {lo:>3}-{hi if hi < 1e5 else "+":<4} share {len(b) / len(sub):.2f} ' + ' '.join(f'{k} {b[k].mean():.3f}' for k in frames))
    print('  overall 1-AUC ' + ' '.join(f'{k} {df[k].mean():.4f}' for k in frames))
