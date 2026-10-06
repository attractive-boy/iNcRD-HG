# -*- coding: utf-8 -*-
"""Round-13 Excel summary: main table (F vs F + EB head, lam 10), per-fold, official protocol, cross-base, lam sensitivity,
strict-pool audit and seed-42 screening of the other candidates."""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'PLMF-MDA_r13_summary.xlsx')
R = lambda *p: os.path.join(HERE, *p)


def load(tag, ds, split, seeds):
    rows = []
    for sd in seeds:
        f = R('results', f'{tag}_{ds}_{split}_seed{sd}.csv')
        if os.path.exists(f):
            d = pd.read_csv(f); d['seed'] = sd; rows.append(d)
    return pd.concat(rows) if rows else None


def main_table():
    X = pd.read_csv(R('per_fold_eb_lam10.csv'))
    out = []
    for ds in ('MDR', 'MDS'):
        for sp in ('5cv', 'single'):
            x = X[(X.ds == ds) & (X.split == sp)]
            out.append(dict(dataset=ds, board='5cv x seeds 42-44' if sp == '5cv' else '80/20 x seeds 0-4', n_folds=len(x),
                            F_auc=x.F_auc.mean(), F_aupr=x.F_aupr.mean(), FEB_auc=x.EB_auc.mean(), FEB_aupr=x.EB_aupr.mean(),
                            d_auc=(x.EB_auc - x.F_auc).mean(), d_aupr=(x.EB_aupr - x.F_aupr).mean(),
                            win_auc=int((x.EB_auc > x.F_auc).sum()), win_aupr=int((x.EB_aupr > x.F_aupr).sum())))
    return pd.DataFrame(out), X


def official():
    out = []
    for ds in ('MDR', 'MDS'):
        for sp, seeds in (('5cv', [42]), ('single', [0, 1, 2])):
            d = load('OFF_FE10', ds, sp, seeds)
            if d is None:
                continue
            out.append(dict(dataset=ds, board=f'official {sp} seeds {seeds}', n=len(d),
                            F_best=f'{d.best_auc.mean():.4f}/{d.best_aupr.mean():.4f}', FEB_best=f'{d.ebp_best_auc.mean():.4f}/{d.ebp_best_aupr.mean():.4f}',
                            F_last=f'{d.last_auc.mean():.4f}/{d.last_aupr.mean():.4f}', FEB_last=f'{d.ebp_last_auc.mean():.4f}/{d.ebp_last_aupr.mean():.4f}',
                            d_best_auc=d.ebp_best_auc.mean() - d.best_auc.mean(), d_best_aupr=d.ebp_best_aupr.mean() - d.best_aupr.mean(),
                            d_last_auc=d.ebp_last_auc.mean() - d.last_auc.mean(), d_last_aupr=d.ebp_last_aupr.mean() - d.last_aupr.mean()))
    return pd.DataFrame(out)


def lam_sens():
    out = []
    for lam in ('1', '3', '10', '30'):
        f = R(f'per_fold_eb_lam{lam}.csv')
        if not os.path.exists(f):
            continue
        X = pd.read_csv(f); X = X[~((X.split == 'single') & (X.seed > 2))]
        for ds in ('MDR', 'MDS'):
            for sp in ('5cv', 'single'):
                x = X[(X.ds == ds) & (X.split == sp)]
                out.append(dict(lam=float(lam), dataset=ds, split=sp, n=len(x), d_auc=(x.EB_auc - x.F_auc).mean(), d_aupr=(x.EB_aupr - x.F_aupr).mean()))
    return pd.DataFrame(out)


def audit():
    out = []
    for sp, seeds in (('5cv', [42, 43, 44]), ('single', [0, 1, 2])):
        for ds in ('MDR', 'MDS'):
            f = load('F', ds, sp, seeds); s = load('FS10', ds, sp, seeds)
            if f is None or s is None:
                continue
            m = s.merge(f, on=['seed', 'fold'], suffixes=('', '_f'))
            out.append(dict(dataset=ds, split=sp, n=len(m), F=f'{m.valsel_auc_f.mean():.4f}/{m.valsel_aupr_f.mean():.4f}',
                            F_strict=f'{m.valsel_auc.mean():.4f}/{m.valsel_aupr.mean():.4f}', F_strict_EB=f'{m.ebp_auc.mean():.4f}/{m.ebp_aupr.mean():.4f}',
                            d_strict_auc=(m.valsel_auc - m.valsel_auc_f).mean(), d_strict_aupr=(m.valsel_aupr - m.valsel_aupr_f).mean(),
                            d_strictEB_vs_F_auc=(m.ebp_auc - m.valsel_auc_f).mean(), d_strictEB_vs_F_aupr=(m.ebp_aupr - m.valsel_aupr_f).mean()))
    return pd.DataFrame(out)


SCREEN = [('FEP', 'post-hoc EB lam 10 (ebp columns)', True), ('FEB10', 'learnable entity intercepts (labelled BCE only)', False),
          ('FEB10all', 'learnable entity intercepts (all losses)', False), ('FH3b', 'HERE in-model EB layer, damped Newton', False),
          ('FHE3b', 'HERE-eval (EB-aware epoch selection) + post-hoc refit', True), ('FDNS1', 'model-mined hard negatives tau 1', False),
          ('FDNS03', 'model-mined hard negatives tau 0.3', False), ('FAD1', 'easy-preferring hard negatives', False),
          ('FSAD1', 'strict pool + easy-preferring negatives', False), ('FEMA9', 'EMA 0.9', False), ('FEMA97', 'EMA 0.97', False),
          ('FZN', 'LayerNorm endpoint / NCN states', False), ('FDO1', 'dropout 0.1', False), ('FHNw2', 'HN weight 2', False),
          ('Fe500', '500 epochs', False), ('FS', 'strict HN/UNL pool (audit)', False), ('Fms1', 'model seed 1', False), ('Fms2', 'model seed 2', False)]


def screening():
    out = []
    for tag, desc, use_ebp in SCREEN:
        for ds in ('MDR', 'MDS'):
            b = load('F', ds, '5cv', [42]); d = load(tag, ds, '5cv', [42])
            if d is None:
                continue
            m = d.merge(b, on=['seed', 'fold'], suffixes=('', '_b'))
            ca, cp = ('ebp_auc', 'ebp_aupr') if use_ebp else ('valsel_auc', 'valsel_aupr')
            out.append(dict(tag=tag, description=desc, dataset=ds, n=len(m), auc=m[ca].mean(), aupr=m[cp].mean(),
                            d_auc=(m[ca] - m.valsel_auc_b).mean(), d_aupr=(m[cp] - m.valsel_aupr_b).mean(),
                            win_auc=int((m[ca] > m.valsel_auc_b).sum()), win_aupr=int((m[cp] > m.valsel_aupr_b).sum())))
    return pd.DataFrame(out)


if __name__ == '__main__':
    mt, pf = main_table()
    with pd.ExcelWriter(OUT) as w:
        mt.to_excel(w, sheet_name='main_valsel', index=False)
        official().to_excel(w, sheet_name='official_protocol', index=False)
        pd.read_csv(R('tables_eb_crossbase_s42.csv')).to_excel(w, sheet_name='cross_base_s42', index=False)
        lam_sens().to_excel(w, sheet_name='lambda_sensitivity', index=False)
        audit().to_excel(w, sheet_name='strict_pool_audit', index=False)
        screening().to_excel(w, sheet_name='screening_s42', index=False)
        pf.to_excel(w, sheet_name='per_fold_lam10', index=False)
    print('wrote', OUT)
    print(mt.round(4).to_string())
