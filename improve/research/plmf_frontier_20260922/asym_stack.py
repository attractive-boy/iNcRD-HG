# -*- coding: utf-8 -*-
"""Does content->latent alignment add beyond the content-kNN baseline on cold drugs?

Extends cold_probe.ColdBank with a `clat` family (content-weighted latent-profile
and ridge content->latent-map columns computed from the *support* adjacency) and
re-runs the same drugcv cross-fitting harness. Baseline to beat: LR on `knn`
features (fit=drugcv), MDR plmf = 0.8137 AUC / 0.8012 AUPR / 0.8338 macroAUC.
"""
import argparse, os, sys, csv, time
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
for p in ('plmf_innov_20260921', 'mphgnn_plmf_20260921', 'dlstmda_strongest_20260920'):
    sys.path.insert(0, os.path.join(BASE, p))
import cold_probe as CP  # noqa


class StackBank(CP.ColdBank):
    def _clat(self, sup, q):
        A = self._A(sup)
        nr = self.D['nr']
        warm = np.where(A.sum(0) > 0)[0]
        r, d = q[:, 0], q[:, 1]
        cols = {}
        if len(warm) < 5:
            for nm in ('tan', 'cb'):
                cols[f'clat_{nm}_lp'] = np.zeros(len(q)); cols[f'clat_{nm}_lp_rpc'] = np.zeros(len(q))
            cols['clat_ridge'] = np.zeros(len(q)); return cols
        U, S, Vt = np.linalg.svd(A[:, warm], full_matrices=False)
        k = min(32, len(S)); Zw = (Vt[:k].T * S[:k]); R = U[:, :k]          # [nw,k],[nr,k]
        F = np.concatenate([_std(self.D['xd']), _std(self.D['xcb'])], axis=1).astype(np.float64)
        Fw_s, Fq_s = _std(F[warm]), _std(F[d])
        Zridge = Ridge(alpha=1.0).fit(Fw_s, Zw).predict(Fq_s)                # [nq,k]
        cols['clat_ridge'] = np.einsum('rk,rk->r', R[r], Zridge)
        for nm, Sfull in (('tan', self.S_tan), ('cb', self.S_cb)):
            W = Sfull[np.ix_(d, warm)]                                       # [nq,nw]
            Zq = (W @ Zw) / np.maximum(W.sum(1, keepdims=True), 1e-8)        # [nq,k]
            allsc = R @ Zq.T                                                 # [nr,nq]
            sc = allsc[r, np.arange(len(q))]
            cols[f'clat_{nm}_lp'] = sc
            # rank of the query among all miRNAs for that drug (popularity-free)
            cols[f'clat_{nm}_lp_rpc'] = (allsc < allsc[r, np.arange(len(q))][None, :]).mean(0)
        return cols

    def transform(self, sup, q, fams):
        base = fams & {'ctx', 'diff', 'cdiff', 'scf', 'align', 'knn', 'mech'}
        out = super().transform(sup, q, base) if base else pd.DataFrame(index=range(len(q)))
        if 'clat' in fams:
            out = pd.concat([out, pd.DataFrame(self._clat(np.asarray(sup, np.int64), np.asarray(q, np.int64)))], axis=1)
        return out.astype(np.float64)


def _std(X):
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True) + 1e-6
    return (X - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MDR'); ap.add_argument('--protocol', default='plmf', choices=['plmf', 'ldo'])
    ap.add_argument('--families', default='knn,clat'); ap.add_argument('--model', default='lr:0.3')
    ap.add_argument('--seeds', default='0,1,2,3,4'); ap.add_argument('--tag', default='stack')
    ap.add_argument('--degmin', type=int, default=11); ap.add_argument('--degmax', type=int, default=20)
    a = ap.parse_args(); fams = set(a.families.split(','))
    D = CP.load_all(a.dataset); cb = StackBank(D)
    A = np.zeros((D['nr'], D['nd']), np.int8); A[D['pos'][:, 0], D['pos'][:, 1]] = 1; ddeg = A.sum(0)
    rows = []
    for seed in [int(s) for s in a.seeds.split(',')]:
        if a.protocol == 'plmf':
            folds = [np.where((ddeg >= a.degmin) & (ddeg <= a.degmax))[0]]
        else:
            from sklearn.model_selection import StratifiedKFold
            bins = np.digitize(ddeg, [2, 6, 21, 101])
            folds = [te for _, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(D['nd']), bins)]
        for fi, test_drugs in enumerate(folds):
            t0 = time.time(); tr, ytr, te, yte = CP.cold_split(D, test_drugs, seed * 10 + fi)
            Xtr, names = CP.train_features(cb, tr, ytr, fams, 'drugcv', seed)
            Xte = cb.transform(tr[ytr == 1], te, fams).values
            Xtr, Xte = CP.prep(Xtr, Xte)
            s, m = CP.fit_predict(a.model, Xtr, ytr, Xte, seed)
            auc, ap_ = roc_auc_score(yte, s), average_precision_score(yte, s); mauc, nd_ = CP.macro_auc(te, yte, s)
            rows.append(dict(seed=seed, auc=auc, aupr=ap_, macro_auc=mauc, ncol=Xtr.shape[1]))
            print(f'  seed {seed}: AUC {auc:.4f} AUPR {ap_:.4f} macroAUC {mauc:.4f} ncol {Xtr.shape[1]} ({time.time()-t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    print(f'\n== {a.tag} {a.dataset} {a.protocol} fams={a.families} {a.model} ==', flush=True)
    print(f'AUC {df.auc.mean():.4f}±{df.auc.std():.4f}  AUPR {df.aupr.mean():.4f}±{df.aupr.std():.4f}  macroAUC {df.macro_auc.mean():.4f}', flush=True)
    out = os.path.join(HERE, 'results', 'asym_stack.csv'); new = not os.path.exists(out)
    with open(out, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['tag', 'dataset', 'protocol', 'families', 'model', 'auc', 'auc_sd', 'aupr', 'aupr_sd', 'macro_auc', 'n_runs'])
        w.writerow([a.tag, a.dataset, a.protocol, a.families, a.model, f'{df.auc.mean():.6f}', f'{df.auc.std():.6f}',
                    f'{df.aupr.mean():.6f}', f'{df.aupr.std():.6f}', f'{df.macro_auc.mean():.6f}', len(df)])


if __name__ == '__main__':
    main()
