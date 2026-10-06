# -*- coding: utf-8 -*-
"""Cross-phenotype side information for COLD drugs on PLMF-MDA.
A drug that is new to the resistance (MDR) database may already have sensitivity (MDS) records
(and vice versa). Reuses cold_probe (protocol plmf: test drugs = degree 11-20) and appends
xds columns computed from the other task's full matrix.  fit=warm cross-fitting for ctx/knn."""
import argparse, os, sys, time, csv
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, 'plmf_frontier_20260922')); sys.path.insert(0, HERE)
import cold_probe as CP  # noqa
from xds_probe import other_matrix  # noqa
from bank import diffusion  # noqa


def xds_cols(Ax, pr, pd_, q, xg):
    r, d = q[:, 0], q[:, 1]
    cols = {'x_pair': Ax[r, d], 'x_deg_r': Ax.sum(1)[r], 'x_deg_d': Ax.sum(0)[d], 'x_shared_r': pr[r], 'x_shared_d': pd_[d]}
    for k, M in xg.items():
        v = M[r, d]; cols[k] = v
        cols[k + '_rz'] = (v - M[r].mean(1)) / (M[r].std(1) + 1e-8)
        cols[k + '_dz'] = (v - M[:, d].mean(0)) / (M[:, d].std(0) + 1e-8)
    return pd.DataFrame(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MDR'); ap.add_argument('--families', default='ctx,knn')
    ap.add_argument('--xds', default='full', choices=['none', 'full', 'pair', 'deg'])
    ap.add_argument('--seeds', default='0,1,2,3,4'); ap.add_argument('--model', default='lr:0.3')
    ap.add_argument('--shuffle_other', action='store_true'); ap.add_argument('--score_col', default='')
    ap.add_argument('--protocol', default='plmf', choices=['plmf', 'ldo'])
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    D = CP.load_all(a.dataset); cb = CP.ColdBank(D)
    Ax, pr, pd_ = other_matrix(a.dataset)
    if a.shuffle_other:
        rng = np.random.RandomState(0); Ax = Ax[rng.permutation(Ax.shape[0])][:, rng.permutation(Ax.shape[1])]
    Dg = diffusion(Ax, 0.3, 0.01)
    xg = dict(x_ppr=Dg['ppr_rd'], x_katz=Dg['katz_rd'], x_pprd=Dg['ppr_dr'], x_srA=cb.bank.sr @ Ax, x_ASd=Ax @ cb.bank.sd)
    A = np.zeros((D['nr'], D['nd']), np.int8); A[D['pos'][:, 0], D['pos'][:, 1]] = 1; ddeg = A.sum(0)
    if a.protocol == 'plmf':
        folds = [np.where((ddeg >= 11) & (ddeg <= 20))[0]]
    else:
        bins = np.digitize(ddeg, [2, 6, 21, 101]); folds = None
    fams = set(a.families.split(','))
    rows = []
    for seed in [int(s) for s in a.seeds.split(',')]:
        if a.protocol == 'ldo':
            rng = np.random.RandomState(seed); fo = np.zeros(D['nd'], int)
            for b in np.unique(bins):
                ids = np.where(bins == b)[0]; fo[ids] = rng.permutation(len(ids)) % 5
            folds = [np.where(fo == f)[0] for f in range(5)]
        for fi, test_drugs in enumerate(folds):
            t0 = time.time()
            tr, ytr, te, yte = CP.cold_split(D, test_drugs, seed * 10 + fi, 'drug', 0)
            n_te_shared = int(pd_[test_drugs].sum()); n_te_xdeg = Ax.sum(0)[test_drugs]
            if a.score_col:
                X = xds_cols(Ax, pr, pd_, te, xg); s = X[a.score_col].values + 1e-6 * np.random.RandomState(seed).rand(len(te))
            else:
                Xtr, names = CP.train_features(cb, tr, ytr, fams, 'warm', seed)
                Xte = cb.transform(tr[ytr == 1], te, fams).values
                if a.xds != 'none':
                    xtr = xds_cols(Ax, pr, pd_, tr, xg); xte = xds_cols(Ax, pr, pd_, te, xg)
                    keep = {'pair': ['x_pair'], 'deg': ['x_deg_r', 'x_deg_d', 'x_shared_r', 'x_shared_d'], 'full': list(xtr.columns)}[a.xds]
                    Xtr = np.concatenate([Xtr, xtr[keep].values], 1); Xte = np.concatenate([Xte, xte[keep].values], 1)
                Xtr, Xte = CP.prep(Xtr, Xte)
                s, _ = CP.fit_predict(a.model, Xtr, ytr, Xte, seed)
            auc = roc_auc_score(yte, s); ap_ = average_precision_score(yte, s); mauc, nd_ = CP.macro_auc(te, yte, s)
            rows.append(dict(seed=seed, fold=fi, auc=auc, aupr=ap_, macro_auc=mauc, n=len(yte)))
            print(f'  seed {seed} fold {fi}: drugs {len(test_drugs)} (in other task: {n_te_shared}, other-deg median {np.median(n_te_xdeg):.0f}) '
                  f'pairs {len(yte)} AUC {auc:.4f} AUPR {ap_:.4f} macroAUC {mauc:.4f} {time.time()-t0:.0f}s', flush=True)
    df = pd.DataFrame(rows)
    tag = a.tag or f'{a.families}+xds:{a.xds}' + ('_shuf' if a.shuffle_other else '') + (f'_col:{a.score_col}' if a.score_col else '')
    print(f'## {tag} {a.dataset} cold-{a.protocol}: AUC {df.auc.mean():.4f}±{df.auc.std():.4f} AUPR {df.aupr.mean():.4f}±{df.aupr.std():.4f} macroAUC {df.macro_auc.mean():.4f}', flush=True)
    out = os.path.join(HERE, 'results', 'cold_xds.csv'); new = not os.path.exists(out)
    with open(out, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new: w.writerow(['tag', 'dataset', 'protocol', 'families', 'xds', 'shuffle', 'score_col', 'auc', 'auc_sd', 'aupr', 'aupr_sd', 'macro_auc', 'n_runs'])
        w.writerow([tag, a.dataset, a.protocol, a.families, a.xds, int(a.shuffle_other), a.score_col, f'{df.auc.mean():.6f}', f'{df.auc.std():.6f}',
                    f'{df.aupr.mean():.6f}', f'{df.aupr.std():.6f}', f'{df.macro_auc.mean():.6f}', len(df)])


if __name__ == '__main__':
    main()
