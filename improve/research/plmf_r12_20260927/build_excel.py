# -*- coding: utf-8 -*-
"""Round-12 Excel summary: main tables from results/ (valsel protocol) + official-protocol (no --val) rows."""
import os, glob
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'PLMF-MDA_r12_summary.xlsx')

TAGS = ['B', 'VG', 'VGHN', 'VGHNUNL', 'VGHNUNLPPR']
OFF_TAGS = ['OFF_B', 'OFF_FIN', 'OFF_FINPPR']
SEEDS5 = [42, 43, 44]
SEEDSS = [0, 1, 2]


def load(tag, ds, split, seeds):
    rows = []
    for sd in seeds:
        f = os.path.join(HERE, 'results', f'{tag}_{ds}_{split}_seed{sd}.csv')
        if os.path.exists(f):
            d = pd.read_csv(f); d['seed'] = sd; rows.append(d)
    return pd.concat(rows) if rows else None


def summary_rows():
    out = []
    for split, seeds in (('5cv', SEEDS5), ('single', SEEDSS)):
        for tag in TAGS:
            for ds in ('MDR', 'MDS'):
                d = load(tag, ds, split, seeds)
                if d is None:
                    continue
                out.append(dict(split=split, seeds=','.join(map(str, seeds)), tag=tag, dataset=ds, n_folds=len(d),
                                best_auc=d.best_auc.mean(), best_aupr=d.best_aupr.mean(),
                                valsel_auc=d.valsel_auc.mean() if 'valsel_auc' in d else np.nan,
                                valsel_aupr=d.valsel_aupr.mean() if 'valsel_aupr' in d else np.nan))
    for split, seeds in (('5cv', [42]), ('single', SEEDSS)):
        for tag in OFF_TAGS:
            for ds in ('MDR', 'MDS'):
                d = load(tag, ds, split, seeds)
                if d is None:
                    continue
                out.append(dict(split=split, seeds=','.join(map(str, seeds)), tag=tag, dataset=ds, n_folds=len(d),
                                best_auc=d.best_auc.mean(), best_aupr=d.best_aupr.mean(),
                                valsel_auc=np.nan, valsel_aupr=np.nan))
    return pd.DataFrame(out)


def paired_rows():
    out = []
    for split, seeds in (('5cv', SEEDS5), ('single', SEEDSS)):
        for tag in TAGS:
            if tag == 'B':
                continue
            for ds in ('MDR', 'MDS'):
                b = load('B', ds, split, seeds); d = load(tag, ds, split, seeds)
                if b is None or d is None:
                    continue
                m = d.merge(b, on=['seed', 'fold'], suffixes=('', '_b'))
                out.append(dict(split=split, tag=tag, dataset=ds, n=len(m),
                                d_best_auc=(m.best_auc - m.best_auc_b).mean(), d_best_aupr=(m.best_aupr - m.best_aupr_b).mean(),
                                win_auc=(m.best_auc > m.best_auc_b).mean(), win_aupr=(m.best_aupr > m.best_aupr_b).mean(),
                                d_valsel_auc=(m.valsel_auc - m.valsel_auc_b).mean(), d_valsel_aupr=(m.valsel_aupr - m.valsel_aupr_b).mean()))
    for split, seeds in (('5cv', [42]), ('single', SEEDSS)):
        for tag in ('OFF_FIN', 'OFF_FINPPR'):
            for ds in ('MDR', 'MDS'):
                b = load('OFF_B', ds, split, seeds); d = load(tag, ds, split, seeds)
                if b is None or d is None:
                    continue
                m = d.merge(b, on=['seed', 'fold'], suffixes=('', '_b'))
                out.append(dict(split='official-' + split, tag=tag, dataset=ds, n=len(m),
                                d_best_auc=(m.best_auc - m.best_auc_b).mean(), d_best_aupr=(m.best_aupr - m.best_aupr_b).mean(),
                                win_auc=(m.best_auc > m.best_auc_b).mean(), win_aupr=(m.best_aupr > m.best_aupr_b).mean(),
                                d_valsel_auc=np.nan, d_valsel_aupr=np.nan))
    return pd.DataFrame(out)


if __name__ == '__main__':
    s = summary_rows(); p = paired_rows()
    folds = []
    for f in sorted(glob.glob(os.path.join(HERE, 'results', '*.csv'))):
        if any(x in f for x in ('scores', 'args', 'curves')):
            continue
        d = pd.read_csv(f); d['run'] = os.path.basename(f).replace('.csv', '')
        folds.append(d)
    folds = pd.concat(folds, ignore_index=True)
    with pd.ExcelWriter(OUT) as xw:
        s.to_excel(xw, 'summary', index=False)
        p.to_excel(xw, 'paired_delta', index=False)
        folds.to_excel(xw, 'per_fold', index=False)
    print('wrote', OUT)
    print(s.round(4).to_string(index=False))
