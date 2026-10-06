"""Aggregate results/*.csv: per (tag, dataset, split) mean over folds x seeds of valsel / best / last AUC & AUPR,
plus paired deltas against a reference tag (same dataset/split/seed/fold).
usage: python summarize.py [--ref TAG] [--tags t1,t2,...] [--split 5cv|single] [--metric valsel|best|last]"""
import argparse, glob, os, sys
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('--ref', default='plain'); ap.add_argument('--tags', default='')
ap.add_argument('--split', default=''); ap.add_argument('--metric', default='valsel'); ap.add_argument('--seeds', default='')
ap.add_argument('--ladder', default='', help='comma list: each tag is compared with the previous one')
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
fs = sorted(glob.glob(os.path.join(HERE, 'results', '*_seed*.csv')))
df = pd.concat([pd.read_csv(f) for f in fs if not f.endswith(('_scores.csv',))], ignore_index=True)
if a.split: df = df[df.split == a.split]
if a.seeds: df = df[df.seed.isin([int(s) for s in a.seeds.split(',')])]
if a.tags: df = df[df.tag.isin(a.tags.split(','))]
m = a.metric
key = ['dataset', 'split', 'seed', 'fold']
def table(ref, tags):
    rows = []
    for (ds, sp), g in df.groupby(['dataset', 'split']):
        r = g[g.tag == ref].set_index(key) if ref else None
        for t in tags:
            x = g[g.tag == t]
            if not len(x): continue
            x = x.set_index(key)
            row = dict(dataset=ds, split=sp, tag=t, n=len(x), seeds=x.reset_index().seed.nunique(),
                       auc=x[f'{m}_auc'].mean(), aupr=x[f'{m}_aupr'].mean(), best_auc=x.best_auc.mean(), last_auc=x.last_auc.mean(),
                       ep=x[f'{m}_epoch'].mean() if f'{m}_epoch' in x else np.nan)
            if r is not None and t != ref:
                j = x.join(r, rsuffix='_ref', how='inner')
                if len(j):
                    da = j[f'{m}_auc'] - j[f'{m}_auc_ref']; dp = j[f'{m}_aupr'] - j[f'{m}_aupr_ref']
                    # per-seed (fold-averaged) wins
                    ps = j.reset_index().groupby('seed').apply(lambda q: pd.Series(dict(da=(q[f'{m}_auc'] - q[f'{m}_auc_ref']).mean(), dp=(q[f'{m}_aupr'] - q[f'{m}_aupr_ref']).mean())))
                    row.update(dAUC=da.mean(), dAUPR=dp.mean(), n_pair=len(j), wins=f'{int((da > 0).sum())}/{len(j)}',
                               seed_wins=f'{int((ps.da > 0).sum())}/{len(ps)}|{int((ps.dp > 0).sum())}/{len(ps)}', d_se=da.std() / np.sqrt(len(da)))
            rows.append(row)
    return pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
if a.ladder:
    L = a.ladder.split(','); out = []
    for i, t in enumerate(L):
        out.append(table(L[i - 1] if i else None, [t]).assign(vs=L[i - 1] if i else ''))
    print(pd.concat(out).to_string(index=False, float_format=lambda v: f'{v:.4f}'))
else:
    tags = a.tags.split(',') if a.tags else sorted(df.tag.unique())
    print(f'metric={m}  ref={a.ref}')
    print(table(a.ref, tags).to_string(index=False, float_format=lambda v: f'{v:.4f}'))
