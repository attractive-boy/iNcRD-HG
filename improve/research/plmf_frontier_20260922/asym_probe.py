# -*- coding: utf-8 -*-
"""Content->collaborative alignment scorers for cold-drug (inductive) MDA.

Validates the R1 thesis in PROPOSAL_NOVEL_TECH.md: on the strict cold-drug
surface, aligning *content* into the *collaborative* space (Shallow-RHS /
DGT-CF prototype style) should beat raw content kNN, while naive
structure+content fusion is negative transfer.

Scorers (all label-free; trained only on warm training drugs):
  content_knn   : Tanimoto/ChemBERTa-similarity weighted warm-drug profile  (baseline)
  svd_map       : truncated SVD of the warm drug-miRNA matrix -> warm drug
                  latent factors; ridge map content features -> latent; apply to cold drug
  lat_profile   : content-weighted average of warm drug latent factors (no regression)
  mi_map        : same ridge map but on the miRNA side (sanity, miRNA is warm)
  combo         : rank-average of content_knn + svd_map

Run with OMP_NUM_THREADS=2. Writes results/asym_probe.csv.
"""
import argparse, os, sys, csv, time
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
for p in ('plmf_innov_20260921', 'mphgnn_plmf_20260921', 'dlstmda_strongest_20260920'):
    sys.path.insert(0, os.path.join(BASE, p))
from cold_probe import load_all, cold_split, macro_auc, tanimoto, cosine  # noqa


def std(X):
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True) + 1e-6
    return (X - mu) / sd


def score_svdmap(A, Z, Fw, Fc, k=32, alpha=1.0, use_content=None, seed=0):
    """A [nr,nd] warm adjacency, Z [nd,k] warm drug latent (V*S from SVD),
    Fw [nd,p] warm drug content, Fc [nc,p] cold drug content.
    Returns cold scores S [nr, nc]."""
    from numpy.linalg import svd
    U, S, Vt = svd(A, full_matrices=False)
    U = U[:, :k]; S = S[:k]; Vt = Vt[:k]
    Zw = (Vt.T * S)                                  # [nd,k] warm drug factors
    R = U                                             # [nr,k] miRNA factors
    Fw_s, Fc_s = std(Fw), std(Fc)
    reg = Ridge(alpha=alpha).fit(Fw_s, Zw)
    Zc = reg.predict(Fc_s)                            # [nc,k] cold drug factors
    return R @ Zc.T                                   # [nr,nc]


def score_knn(A, S, cold_rows, warm=True):
    num = A @ S.T                                     # [nr, nd]
    den = np.maximum(S.sum(1), 1e-8)
    return num / den[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MDR'); ap.add_argument('--protocol', default='plmf', choices=['plmf', 'ldo'])
    ap.add_argument('--seeds', default='0,1,2,3,4'); ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--content', default='both', choices=['morgan', 'cb', 'both'])
    ap.add_argument('--tag', default='asym')
    a = ap.parse_args()
    D = load_all(a.dataset)
    nr, nd = D['nr'], D['nd']
    S_tan = tanimoto(D['xd']); S_cb = cosine(D['xcb'])
    if a.content == 'morgan':
        F = D['xd']
    elif a.content == 'cb':
        F = D['xcb']
    else:
        F = np.concatenate([std(D['xd']), std(D['xcb'])], axis=1).astype(np.float32)
    A_full = np.zeros((nr, nd), np.int8); A_full[D['pos'][:, 0], D['pos'][:, 1]] = 1
    ddeg = A_full.sum(0)

    rows = []
    for seed in [int(s) for s in a.seeds.split(',')]:
        if a.protocol == 'plmf':
            folds = [np.where((ddeg >= 11) & (ddeg <= 20))[0]]
        else:
            bins = np.digitize(ddeg, [2, 6, 21, 101])
            from sklearn.model_selection import StratifiedKFold
            folds = [te for _, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(nd), bins)]
        for fi, test_drugs in enumerate(folds):
            t0 = time.time()
            tr, ytr, te, yte = cold_split(D, test_drugs, seed * 10 + fi)
            sup = tr[ytr == 1]
            A = np.zeros((nr, nd), np.float32); A[sup[:, 0], sup[:, 1]] = 1
            warm = np.where(A.sum(0) > 0)[0]
            cold = np.array(sorted(set(test_drugs.tolist())))
            r = te[:, 0]; d = te[:, 1]
            scores = {}

            # content kNN baselines
            for nm, S in (('tan', S_tan), ('cb', S_cb)):
                num = A @ S.T; den = np.maximum(S.sum(1), 1e-8)
                scores[f'content_knn_{nm}'] = (num / den[None, :])[r, d]

            # content -> latent SVD map (warm drugs only for SVD/regression)
            U, Sv, Vt = np.linalg.svd(A[:, warm], full_matrices=False)
            k = min(a.k, len(Sv))
            Zw = (Vt[:k].T * Sv[:k]); R = U[:, :k]
            Fw_s, Fc_s = std(F[warm]), std(F[cold])
            for alpha in (0.3, 1.0, 3.0):
                Zc = Ridge(alpha=alpha).fit(Fw_s, Zw).predict(Fc_s)
                full = np.zeros((nr, nd), np.float32); full[:, cold] = R @ Zc.T
                scores[f'svd_map_a{alpha}'] = full[r, d]
            # content-weighted latent profile (neighbour transfer), tan + cb, dense and top-k
            def lat_prof(S):
                W = S[np.ix_(cold, warm)]
                num = W @ Zw
                den = np.maximum(W.sum(1, keepdims=True), 1e-8)
                full = np.zeros((nr, nd), np.float32); full[:, cold] = R @ (num / den).T
                return full[r, d]
            for nm, S in (('tan', S_tan), ('cb', S_cb)):
                scores[f'lat_profile_{nm}'] = lat_prof(S)
                for kk in (5, 20):
                    Sk = S.copy()
                    thr = np.sort(Sk, axis=1)[:, -kk][:, None]; Sk = np.where(Sk >= thr, Sk, 0)
                    W = Sk[np.ix_(cold, warm)]; den = np.maximum(W.sum(1, keepdims=True), 1e-8)
                    full = np.zeros((nr, nd), np.float32); full[:, cold] = R @ ((W @ Zw) / den).T
                    scores[f'lat_profile_{nm}{kk}'] = full[r, d]

            # rank-average combos
            def rk(x): return rankdata(x) / len(x)
            lp = scores['lat_profile_tan']
            scores['combo_knn+lat'] = rk(scores['content_knn_tan']) + rk(lp)
            scores['combo_knn+lat+cb'] = rk(scores['content_knn_tan']) + rk(lp) + rk(scores['lat_profile_cb'])
            # miRNA train-degree as a legitimately warm signal
            mird = A.sum(1)
            scores['knn+mirdeg'] = rk(scores['content_knn_tan']) + 0.5 * rk(np.array([mird[x] for x in r]))

            for nm, s in scores.items():
                auc = roc_auc_score(yte, s); ap_ = average_precision_score(yte, s); mauc, nd_ = macro_auc(te, yte, s)
                rows.append(dict(seed=seed, fold=fi, scorer=nm, auc=auc, aupr=ap_, macro_auc=mauc, n_macro=nd_,
                                 n_cold=len(cold), n_pairs=len(yte)))
            print(f'  seed {seed} fold {fi}: cold drugs {len(cold)} pairs {len(yte)} '
                  + ' '.join(f'{nm}:{scores[nm].shape} ' for nm in ['content_knn_tan'])
                  + f'| knn_auc {roc_auc_score(yte, scores["content_knn_tan"]):.4f} '
                    f'lat_auc {roc_auc_score(yte, scores["lat_profile_tan"]):.4f} '
                    f'combo_auc {roc_auc_score(yte, scores["combo_knn+lat"]):.4f} {time.time()-t0:.0f}s', flush=True)

    df = pd.DataFrame(rows)
    print(f'\n== {a.tag} {a.dataset} {a.protocol} content={a.content} ==', flush=True)
    summary = []
    for nm in df.scorer.unique():
        g = df[df.scorer == nm]
        summary.append((nm, g.auc.mean(), g.auc.std(), g.aupr.mean(), g.aupr.std(), g.macro_auc.mean()))
        print(f'{nm:22s} AUC {g.auc.mean():.4f}±{g.auc.std():.4f}  AUPR {g.aupr.mean():.4f}±{g.aupr.std():.4f}  macroAUC {g.macro_auc.mean():.4f}', flush=True)
    out = os.path.join(HERE, 'results', 'asym_probe.csv'); new = not os.path.exists(out)
    with open(out, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['tag', 'dataset', 'protocol', 'content', 'scorer', 'auc', 'auc_sd', 'aupr', 'aupr_sd', 'macro_auc', 'n_runs'])
        for nm, m1, s1, m2, s2, m3 in summary:
            w.writerow([a.tag, a.dataset, a.protocol, a.content, nm, f'{m1:.6f}', f'{s1:.6f}', f'{m2:.6f}', f'{s2:.6f}', f'{m3:.6f}', len(df[df.scorer == nm])])


if __name__ == '__main__':
    main()
