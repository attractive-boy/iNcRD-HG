# -*- coding: utf-8 -*-
"""Where does the round-12 stack gain?  Positive-side ranking loss of the test positives, binned by the
miRNA / drug degree in the fold's message graph.  Compares a reference tag with a candidate tag on the
same fold (per-pair scores saved by run_r12.py in results/{tag}_{DS}_{split}_seed{seed}_scores.csv)."""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, 'mphgnn_plmf_20260921'))
from plmf_splits import get_split, load_dataset  # noqa: E402

EDGES = [-0.5, 0.5, 1.5, 3.5, 7.5, 15.5, 10**9]
LABELS = ['0', '1', '2-3', '4-7', '8-15', '16+']


def degrees(dataset, split, seed, fold):
    mir, dru, pos, neg = load_dataset(dataset)
    tr, ytr, te, yte = get_split(dataset, split, fold, seed)
    A = np.zeros((len(mir), len(dru))); A[tr[ytr == 1][:, 0], tr[ytr == 1][:, 1]] = 1
    return A.sum(1), A.sum(0)


def main():
    ds, split, seed = sys.argv[1], sys.argv[2], sys.argv[3]
    ref, cand = sys.argv[4], sys.argv[5]
    seeds = [int(s) for s in seed.split(',')]
    folds = range(5) if split == '5cv' else range(1)
    rows = []
    for sd in seeds:
        for f in folds:
            dm, dd = degrees(ds, split, sd, f)
            fr = pd.read_csv(os.path.join(HERE, 'results', f'{ref}_{ds}_{split}_seed{sd}_scores.csv'))
            fc = pd.read_csv(os.path.join(HERE, 'results', f'{cand}_{ds}_{split}_seed{sd}_scores.csv'))
            fr, fc = fr[fr.fold == f], fc[fc.fold == f]
            for pos_side, degs in (('m', dm), ('d', dd)):
                for tag, df in ((ref, fr), (cand, fc)):
                    p = df[df.y == 1].copy()
                    p['deg'] = degs[p[pos_side].values]
                    p['bin'] = pd.cut(p.deg, bins=EDGES, labels=LABELS)
                    sn = np.sort(df[df.y == 0].score.values)
                    p['rl'] = [(sn > s).mean() for s in p.score.values]
                    g = p.groupby('bin', observed=True).agg(n=('rl', 'size'), rl=('rl', 'mean')).reset_index()
                    g['tag'] = tag; g['side'] = pos_side
                    rows.append(g)
    T = pd.concat(rows)
    for side in ('m', 'd'):
        print(f'== {ds} {split} seeds={seeds} -- positive-side rank loss by {side} degree')
        for b in LABELS:
            a = T[(T.side == side) & (T.bin == b) & (T.tag == ref)]
            c = T[(T.side == side) & (T.bin == b) & (T.tag == cand)]
            if len(a) == 0 or len(c) == 0:
                continue
            print(f'  {b:>5}  n={int(a.n.sum()):6d}  {ref:>10} {a.rl.mean():.4f}  {cand:>10} {c.rl.mean():.4f}  delta {c.rl.mean() - a.rl.mean():+.4f}')


if __name__ == '__main__':
    main()
