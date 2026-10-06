# -*- coding: utf-8 -*-
"""Paired comparison of run_int.py results.
usage: python summarize.py --ref pair --tags a,b,c [--splits 5cv:42,5cv:43] [--metric valsel]
Prints per-dataset mean AUC/AUPR, paired delta vs --ref (same fold & seed), wins/n."""
import argparse, glob, os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--ref', default='pair')
ap.add_argument('--tags', default='')
ap.add_argument('--splits', default='5cv:42')
ap.add_argument('--metric', default='valsel')
ap.add_argument('--ladder', default='', help='comma list; each rung is compared with the previous one')
ap.add_argument('--dir', default=os.path.join(HERE, 'results'))
a = ap.parse_args()


def load(tag, ds):
    out = []
    for sp in a.splits.split(','):
        split, seed = sp.split(':')
        f = os.path.join(a.dir, f'{tag}_{ds}_{split}_seed{seed}.csv')
        if os.path.exists(f):
            d = pd.read_csv(f); d['key'] = d['split'] + '_' + d['seed'].astype(str) + '_' + d['fold'].astype(str); out.append(d)
    return pd.concat(out) if out else None


def cmp(tag, ref, ds):
    d, r = load(tag, ds), load(ref, ds)
    if d is None:
        return None
    m = a.metric
    res = dict(n=len(d), auc=d[f'{m}_auc'].mean(), aupr=d[f'{m}_aupr'].mean())
    if r is not None and tag != ref:
        j = d.merge(r, on='key', suffixes=('', '_r'))
        if len(j):
            da = j[f'{m}_auc'] - j[f'{m}_auc_r']; dp = j[f'{m}_aupr'] - j[f'{m}_aupr_r']
            res.update(dauc=da.mean(), daupr=dp.mean(), se=da.std(ddof=1) / np.sqrt(len(j)) if len(j) > 1 else np.nan,
                       win=int((da > 0).sum()), winp=int((dp > 0).sum()), nj=len(j))
    return res


def fmt(tag, ds, res):
    s = f'{ds} {tag:28s} n={res["n"]:2d} {res["auc"]:.4f}/{res["aupr"]:.4f}'
    if 'dauc' in res:
        s += f'  d={res["dauc"]:+.4f}/{res["daupr"]:+.4f} (se {res["se"]:.4f}, win {res["win"]}/{res["nj"]} | {res["winp"]}/{res["nj"]})'
    return s


for ds in ('MDR', 'MDS'):
    if a.ladder:
        rungs = a.ladder.split(',')
        for i, t in enumerate(rungs):
            res = cmp(t, rungs[i - 1] if i else t, ds)
            if res:
                print(fmt(t, ds, res))
    else:
        tags = [a.ref] + [t for t in a.tags.split(',') if t and t != a.ref]
        if not a.tags:
            fs = sorted(glob.glob(os.path.join(a.dir, f'*_{ds}_*.csv')))
            tags = [a.ref] + sorted({os.path.basename(f).split(f'_{ds}_')[0] for f in fs if not f.endswith(('_scores.csv',))} - {a.ref})
        for t in tags:
            res = cmp(t, a.ref, ds)
            if res:
                print(fmt(t, ds, res))
    print()
