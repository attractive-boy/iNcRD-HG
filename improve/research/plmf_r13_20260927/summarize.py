# -*- coding: utf-8 -*-
"""Summarize round-13 results: per-tag means over folds/seeds, paired deltas vs a reference tag.

usage: python summarize.py <split> <seed1,seed2,...> <ref_tag> [tags...]
   a tag written as TAG:ebp reads the post-hoc entity-bias columns (ebp_auc / ebp_aupr) of TAG as its valsel metric
   e.g. python summarize.py 5cv 42 F FEP:ebp FEB10 FEB3 FDNS1
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag, ds, split, seeds):
    tag, _, col = tag.partition(':')
    rows = []
    for sd in seeds:
        f = os.path.join(HERE, 'results', f'{tag}_{ds}_{split}_seed{sd}.csv')
        if os.path.exists(f):
            d = pd.read_csv(f); d['seed'] = sd
            if col == 'ebp':
                d['valsel_auc'], d['valsel_aupr'] = d['ebp_auc'], d['ebp_aupr']
                d['best_auc'], d['best_aupr'] = np.nan, np.nan
            rows.append(d)
    return pd.concat(rows) if rows else None


def main():
    split = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2].split(',')]
    ref = sys.argv[3]
    tags = sys.argv[4:]
    for ds in ('MDR', 'MDS'):
        print(f'== {ds} {split} seeds={seeds}')
        base = load(ref, ds, split, seeds)
        for tag in [ref] + tags:
            d = load(tag, ds, split, seeds)
            if d is None:
                print(f'  {tag:14s} (no results)'); continue
            line = f'  {tag:14s} n={len(d):2d} valsel {d.valsel_auc.mean():.4f}/{d.valsel_aupr.mean():.4f} | best {d.best_auc.mean():.4f}/{d.best_aupr.mean():.4f}'
            if base is not None and tag != ref:
                m = d.merge(base, on=['seed', 'fold'], suffixes=('', '_b'))
                if len(m):
                    line += (f' | d(valsel) {m.valsel_auc.mean() - m.valsel_auc_b.mean():+.4f}/{m.valsel_aupr.mean() - m.valsel_aupr_b.mean():+.4f}'
                             f' win {int((m.valsel_auc > m.valsel_auc_b).sum())}/{int((m.valsel_aupr > m.valsel_aupr_b).sum())} of {len(m)}'
                             f' | d(best) {m.best_auc.mean() - m.best_auc_b.mean():+.4f}/{m.best_aupr.mean() - m.best_aupr_b.mean():+.4f}')
            print(line)


if __name__ == '__main__':
    main()
