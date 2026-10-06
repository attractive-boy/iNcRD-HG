"""Where do the gains land?  Positive-side ranking loss (fraction of same-fold test negatives scored above the
positive) bucketed by the miRNA's support degree in the fold's training-positive graph.  Uses results/*_scores.csv
(valsel-epoch scores).  usage: python gain_analysis.py --split 5cv --seed 42 --tags plain,lqo,pair,pair_prov,full"""
import argparse, os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'mphgnn_plmf_20260921'))
from plmf_splits import get_split
ap = argparse.ArgumentParser(); ap.add_argument('--split', default='5cv'); ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--tags', default='plain,lqo,pair,pair_prov,full'); a = ap.parse_args()
B = [(0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 10 ** 6)]
for ds in ('MDR', 'MDS'):
    out = {}
    for tag in a.tags.split(','):
        f = os.path.join(HERE, 'results', f'{tag}_{ds}_{a.split}_seed{a.seed}_scores.csv')
        if not os.path.exists(f):
            continue
        sc = pd.read_csv(f); rows = []
        for fold, g in sc.groupby('fold'):
            tr, ytr, te, yte = get_split(ds, a.split, int(fold), a.seed)
            deg = np.bincount(tr[ytr == 1][:, 0], minlength=10 ** 4)
            neg = np.sort(g[g.y == 0].score.values)
            pos = g[g.y == 1]
            loss = 1.0 - np.searchsorted(neg, pos.score.values, side='left') / len(neg)   # fraction of negatives above
            rows.append(pd.DataFrame(dict(deg=deg[pos.m.values], loss=loss)))
        r = pd.concat(rows)
        r['bucket'] = pd.cut(r.deg, [-1, 0, 1, 3, 7, 15, 10 ** 6], labels=['0', '1', '2-3', '4-7', '8-15', '16+'])
        out[tag] = r.groupby('bucket').loss.mean()
        out.setdefault('share', r.groupby('bucket').size() / len(r))
    d = pd.DataFrame(out)
    print(f'\n{ds} {a.split} seed{a.seed}: positive-side mean ranking loss by miRNA support degree')
    print(d.to_string(float_format=lambda v: f'{v:.4f}'))
    tags = [t for t in a.tags.split(',') if t in d]
    if len(tags) >= 2:
        print('relative loss reduction first->last:', ((d[tags[0]] - d[tags[-1]]) / d[tags[0]]).round(3).to_dict())
