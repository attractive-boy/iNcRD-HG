# -*- coding: utf-8 -*-
"""Summarize round-11 results: per-tag means over folds/seeds, paired deltas vs a reference tag.

usage: python summarize.py <split> <seed1,seed2,...> [ref_tag] [tags...]
   e.g. python summarize.py 5cv 42,43,44 B PPRn HN PPRnHN UNL PPRnHNUNL
"""
import os, sys, glob
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag, ds, split, seeds):
    rows = []
    for sd in seeds:
        f = os.path.join(HERE, 'results', f'{tag}_{ds}_{split}_seed{sd}.csv')
        if os.path.exists(f):
            d = pd.read_csv(f); d['seed'] = sd; rows.append(d)
    return pd.concat(rows) if rows else None


def main():
    split = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2].split(',')]
    ref = sys.argv[3] if len(sys.argv) > 3 else None
    tags = sys.argv[4:] if len(sys.argv) > 4 else ['B', 'PPRn', 'HN', 'PPRnHN', 'UNL', 'PPRnHNUNL']
    if not tags and ref:
        tags = [ref]
    for ds in ('MDR', 'MDS'):
        print(f'== {ds} {split} seeds={seeds}')
        base = load(ref, ds, split, seeds) if ref else None
        for tag in tags:
            d = load(tag, ds, split, seeds)
            if d is None:
                print(f'  {tag:12s} (no results)'); continue
            line = f'  {tag:12s} n={len(d):2d} AUC {d.best_auc.mean():.4f} AUPR {d.best_aupr.mean():.4f} | valsel {d.valsel_auc.mean():.4f}/{d.valsel_aupr.mean():.4f}'
            if base is not None and tag != ref:
                # paired by (seed, fold)
                m = d.merge(base, on=['seed', 'fold'], suffixes=('', '_b'))
                if len(m):
                    line += (f' | d(best) {m.best_auc.mean() - m.best_auc_b.mean():+.4f}/{m.best_aupr.mean() - m.best_aupr_b.mean():+.4f}'
                             f' winAUC {(m.best_auc > m.best_auc_b).mean():.2f} winAUPR {(m.best_aupr > m.best_aupr_b).mean():.2f}'
                             f' | d(valsel) {m.valsel_auc.mean() - m.valsel_auc_b.mean():+.4f}/{m.valsel_aupr.mean() - m.valsel_aupr_b.mean():+.4f}')
            print(line)


if __name__ == '__main__':
    main()
