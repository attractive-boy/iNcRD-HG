# -*- coding: utf-8 -*-
"""Summarise a pilot run set: per-config means and paired deltas vs plain, per split.
Usage: python summarize.py [prefix]   (prefix default P_; use K_ for the kNN-control grid)"""
import glob, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = sys.argv[1] if len(sys.argv) > 1 else 'P_'
NAMES = ['plain', 'M1', 'M2', 'M3', 'M1M2', 'M1M3', 'M2M3', 'full', 'nomask']
LABEL = {'plain': 'plain', 'M1': '+ M1 QMP', 'M2': '+ M2 SP-RNCN', 'M3': '+ M3 MC-EASE',
         'M1M2': '+ M1+M2', 'M1M3': '+ M1+M3', 'M2M3': '+ M2+M3 (no M1)',
         'full': '+ M1+M2+M3 (full)', 'nomask': 'same channels, no masking (control)'}
CONFIGS = [PREFIX + n for n in NAMES]


def load():
    rows = []
    for f in glob.glob(os.path.join(HERE, 'results', f'{PREFIX}*.csv')):
        if f.endswith('_scores.csv'):
            continue
        d = pd.read_csv(f)
        d['file'] = os.path.basename(f)
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d['key'] = d['split'].astype(str) + '|' + d['split_seed'].astype(str) + '|' + d['model_seed'].astype(str)
    return d


def main():
    d = load()
    print(f'# own MiDrug pilot [{PREFIX}] -- {len(d)} runs, {d.key.nunique()} split/model cells\n')
    print('| config | units | valsel AUC | valsel AUPR | last AUC | last AUPR |')
    print('|---|---|---|---|---|---|')
    for c in CONFIGS:
        x = d[d.tag == c]
        if not len(x):
            continue
        print(f'| {LABEL[c[len(PREFIX):]]} | {len(x)} | {x.valsel_auc.mean():.4f} | {x.valsel_aupr.mean():.4f}'
              f' | {x.last_auc.mean():.4f} | {x.last_aupr.mean():.4f} |')
    print()
    print('## paired deltas vs plain (matched split/model seed)\n')
    print('| config | dAUC valsel | dAUPR valsel | wins AUC | dAUC last | dAUPR last | wins AUC last |')
    print('|---|---|---|---|---|---|---|')
    base = d[d.tag == PREFIX + 'plain'].set_index('key')
    for c in CONFIGS[1:]:
        x = d[d.tag == c].set_index('key')
        keys = sorted(set(base.index) & set(x.index))
        if not keys:
            continue
        da = np.array([x.loc[k, 'valsel_auc'] - base.loc[k, 'valsel_auc'] for k in keys])
        dp = np.array([x.loc[k, 'valsel_aupr'] - base.loc[k, 'valsel_aupr'] for k in keys])
        dla = np.array([x.loc[k, 'last_auc'] - base.loc[k, 'last_auc'] for k in keys])
        dlp = np.array([x.loc[k, 'last_aupr'] - base.loc[k, 'last_aupr'] for k in keys])
        print(f'| {LABEL[c[len(PREFIX):]]} | {da.mean():+.4f} (se {da.std(ddof=1)/np.sqrt(len(keys)):.4f}) | {dp.mean():+.4f}'
              f' | {int((da > 0).sum())}/{len(keys)} | {dla.mean():+.4f} | {dlp.mean():+.4f} | {int((dla > 0).sum())}/{len(keys)} |')
    print()
    print('## per-cell valsel AUC / AUPR\n')
    piv = d.pivot_table(index='key', columns='tag', values=['valsel_auc', 'valsel_aupr'])
    cols = [t for t in CONFIGS if t in piv['valsel_auc'].columns]
    print('| split|seed|model | ' + ' | '.join(cols) + ' |')
    print('|---|' + '---|' * len(cols))
    for k in sorted(piv.index):
        vals = [f"{piv.loc[k, ('valsel_auc', c)]:.4f}/{piv.loc[k, ('valsel_aupr', c)]:.4f}" for c in cols]
        print(f'| {k} | ' + ' | '.join(vals) + ' |')


if __name__ == '__main__':
    main()
