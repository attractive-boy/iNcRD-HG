"""usage: python make_tables.py METRIC ROW:REF ...   METRIC in valsel|best ; paired deltas vs REF on four boards"""
import sys, glob, os, numpy as np, pandas as pd
metric = sys.argv[1]; rows = [x.split(':') for x in sys.argv[2:]]
D = pd.concat([pd.read_csv(f) for f in glob.glob('results/*.csv') if not f.endswith('_scores.csv')])
BOARDS = [('MDR', '5cv', [42, 43, 44]), ('MDS', '5cv', [42, 43, 44]), ('MDR', 'single', [0, 1, 2, 3, 4]), ('MDS', 'single', [0, 1, 2, 3, 4])]
def get(tag, ds, sp, seeds):
    x = D[(D.tag == tag) & (D.dataset == ds) & (D.split == sp) & (D.seed.isin(seeds))]
    return x.set_index(['seed', 'fold'])[[f'{metric}_auc', f'{metric}_aupr']] if len(x) else None
hdr = '| config | ' + ' | '.join(f'{ds} {"5cv×3" if sp == "5cv" else "80/20×5"} | Δ' for ds, sp, _ in BOARDS) + ' |'
print(hdr); print('|' + '---|' * (1 + 2 * len(BOARDS)))
for tag, ref in rows:
    cells = []
    for ds, sp, seeds in BOARDS:
        x = get(tag, ds, sp, seeds); r = get(ref, ds, sp, seeds) if ref != '-' else None
        if x is None: cells += ['–', '–']; continue
        cells.append(f'{x.iloc[:, 0].mean():.4f} / {x.iloc[:, 1].mean():.4f}' + ('' if len(x) == (15 if sp == '5cv' else 5) else f' (n={len(x)})'))
        if r is None: cells.append('–'); continue
        i = x.index.intersection(r.index); da = x.iloc[:, 0][i] - r.iloc[:, 0][i]; dp = x.iloc[:, 1][i] - r.iloc[:, 1][i]
        cells.append(f'{da.mean():+.4f} / {dp.mean():+.4f} ({(da > 0).sum()}·{(dp > 0).sum()}/{len(i)})')
    print(f'| {tag} vs {ref} | ' + ' | '.join(cells) + ' |')
