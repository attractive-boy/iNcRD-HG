# -*- coding: utf-8 -*-
"""Where do the gains land?  Positive-side ranking loss (fraction of test negatives scored above each
test positive) bucketed by the miRNA's training support degree; paired over the same test pairs."""
import sys, os, argparse
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'mphgnn_plmf_20260921'))
from plmf_splits import get_split
ap = argparse.ArgumentParser(); ap.add_argument('--tags', default='plain,full_ext'); ap.add_argument('--split', default='5cv'); ap.add_argument('--seed', type=int, default=42)
a = ap.parse_args()
BK = [(0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 999)]
for ds in ('MDR', 'MDS'):
    rows = []
    for tag in a.tags.split(','):
        sc = pd.read_csv(os.path.join(HERE, 'results', f'{tag}_{ds}_{a.split}_seed{a.seed}_scores.csv'))
        for f, g in sc.groupby('fold'):
            tr, ytr, te, yte = get_split(ds, a.split, int(f), a.seed)
            deg = np.bincount(tr[ytr == 1, 0], minlength=2000)
            neg = np.sort(g.score[g.y == 0].values)
            p = g[g.y == 1]
            loss = 1 - np.searchsorted(neg, p.score.values) / len(neg)
            dm = deg[p.m.values]
            for lo, hi in BK:
                sel = (dm >= lo) & (dm <= hi)
                rows.append(dict(tag=tag, fold=f, bucket=f'{lo}-{hi}' if hi < 999 else f'{lo}+', share=sel.mean(), loss=loss[sel].mean(), n=sel.sum()))
    d = pd.DataFrame(rows).groupby(['bucket', 'tag'])[['share', 'loss']].mean().unstack('tag')
    print(ds); print(d.round(3))
