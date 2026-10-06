# -*- coding: utf-8 -*-
"""Clean (label-leak-free) AUPR ceiling probe.

Enriches the 33-column graph-context pair features with global diffusion /
spectral features (personalized PageRank, Katz, path-7) computed from the
TRAINING-positive bipartite graph using a fast odd-walk power series
(A^{2k+1} = A (A^T A)^k), then fits a strong single classifier
(HistGradientBoosting or TabICLv2) with the same 5-fold cross-fitting of
training-pair features as features.py::crossfit_features (test uses the full
training-positive support).  No test labels are ever used.
"""
import os, sys, hashlib, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, 'mphgnn_plmf_20260921'))
sys.path.insert(0, os.path.join(BASE, 'dlstmda_strongest_20260920'))
from plmf_splits import get_split  # noqa
import features as F33  # noqa

DIFF = ((0.3, 0.01), (0.6, 0.05))
KMAX = 12


def _diffusion(A, alpha, beta):
    """Return odd-length walk aggregates on the bipartite graph A[nr,nd]."""
    nr, nd = A.shape
    dsum = np.maximum(A.sum(1, keepdims=True), 1)
    ddsum = np.maximum(A.sum(0, keepdims=True), 1)
    An = A / dsum                        # miRNA -> drug transition
    And = (A / ddsum).T                  # drug -> miRNA transition
    G = And @ An                         # drug x drug (2-step back to a drug)
    Gs = An @ And                        # miRNA x miRNA
    ppr = np.zeros((nr, nd)); katz = np.zeros((nr, nd))
    p = An.copy()                        # length-1 walks (miRNA->drug)
    q = And.copy()                       # length-1 walks (drug->miRNA)
    for k in range(KMAX):
        c_a = alpha * (1 - alpha) ** (2 * k + 1)
        c_b = beta ** (2 * k + 1)
        ppr += (c_a + c_a * (1 - alpha)) * p if False else c_a * p
        katz += c_b * p
        p = p @ G
        # drug->miRNA direction
        q = q @ Gs
    # drug->miRNA ppr/katz (transpose roles)
    ppr_d = np.zeros((nd, nr)); katz_d = np.zeros((nd, nr))
    q2 = And.copy()
    p2 = An.copy()
    for k in range(KMAX):
        c_a = alpha * (1 - alpha) ** (2 * k + 1)
        c_b = beta ** (2 * k + 1)
        ppr_d += c_a * q2
        katz_d += c_b * q2
        q2 = q2 @ Gs
    return dict(ppr_rd=ppr, katz_rd=katz, ppr_dr=ppr_d.T, katz_dr=katz_d.T,
                path7=(A @ np.linalg.matrix_power(G, 3)))


class Enricher:
    def __init__(self, xr, xd, sr, sd):
        self.enc = F33.GraphPairEncoder(xr, xd, sr, sd)
        self.nr, self.nd = xr.shape[0], xd.shape[0]
        self.sr, self.sd = sr.astype(np.float64), sd.astype(np.float64)
        self.cache = {}

    def _ctx33(self, sup, q):
        return self.enc.transform(q.T, sup.T).values.astype(np.float32)

    def _global(self, sup):
        k = hashlib.md5(np.sort(sup[0] * 100000 + sup[1]).tobytes()).hexdigest()
        if k in self.cache:
            return self.cache[k]
        A = np.zeros((self.nr, self.nd), np.float64); A[sup[0], sup[1]] = 1.0
        out = {}
        for al, be in DIFF:
            d = _diffusion(A, al, be)
            for key in ('ppr_rd', 'katz_rd', 'ppr_dr', 'katz_dr', 'path7'):
                out[f'{key}_{int(al*100)}_{int(be*1000)}'] = d[key]
        out['srA'] = self.sr @ A
        out['ASd'] = A @ self.sd
        out['srASd'] = self.sr @ A @ self.sd
        self.cache[k] = out
        return out

    def transform(self, sup, q):
        base = self._ctx33(sup, q)
        gm = self._global(sup.T)
        if not gm:
            return base
        r, d = q[:, 0], q[:, 1]
        cols = []
        for k, D in gm.items():
            v = D[r, d]
            cols.append(v)
            rm = D.mean(1, keepdims=True); rs = D.std(1, keepdims=True) + 1e-8
            cols.append((v - D[r].mean(1)) / (D[r].std(1) + 1e-8))          # per-miRNA z
            cols.append((D[r] < v[:, None]).mean(1))                         # per-miRNA percentile
            cm = D.mean(0, keepdims=True); cs = D.std(0, keepdims=True) + 1e-8
            cols.append((v - D[:, d].mean(0)) / (D[:, d].std(0) + 1e-8))     # per-drug z
        extra = np.stack(cols, 1).astype(np.float32)
        return np.concatenate([base, extra], 1)


def crossfit(enr, tr, ytr, te, seed):
    Xtr = None
    for fit, qy in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, ytr.astype(int)):
        sup = tr[fit[ytr[fit] == 1]]
        f = enr.transform(sup, tr[qy])
        if Xtr is None:
            Xtr = np.zeros((len(ytr), f.shape[1]), np.float32)
        Xtr[qy] = f
    Xte = enr.transform(tr[ytr == 1], te)
    Xtr = np.log1p(np.abs(Xtr)) * np.sign(Xtr)
    Xte = np.log1p(np.abs(Xte)) * np.sign(Xte)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd, Xtr.shape[1]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MDR')
    ap.add_argument('--split', default='single')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--enrich', type=int, default=1)
    ap.add_argument('--tabicl', type=int, default=0)
    ap.add_argument('--folds', default='', help='comma list, default all')
    a = ap.parse_args()
    import csv as _csv

    g = np.load(os.path.join(BASE, 'mphgnn_plmf_20260921', 'cache', f'graph_{a.dataset}.npz'), allow_pickle=True)
    xr = g['mirna_kmer'].astype(np.float32); xd = g['drug_x'].astype(np.float32)
    nr, nd = xr.shape[0], xd.shape[0]
    z = np.zeros((nr, nr), np.float32); z2 = np.zeros((nd, nd), np.float32)
    ux = F33.GraphPairEncoder(xr, xd, z, z2).ur
    sr = F33.knn_adjacency((ux @ ux.T).astype(np.float32), 10)
    un = F33.GraphPairEncoder(xr, xd, z, z2).ud
    sd = F33.knn_adjacency((un @ un.T).astype(np.float32), 10)
    enr = Enricher(xr, xd, sr, sd)
    if not a.enrich:
        enr._global = lambda sup: {}

    folds = [0] if a.split == 'single' else list(range(5))
    if a.folds:
        folds = [int(x) for x in a.folds.split(',')]
    outdir = os.path.join(HERE, 'results')
    os.makedirs(outdir, exist_ok=True)
    csvp = os.path.join(outdir, 'enrich_probe.csv')
    newfile = not os.path.exists(csvp)
    fh = open(csvp, 'a', newline=''); w = _csv.writer(fh)
    if newfile:
        w.writerow(['dataset', 'split', 'seed', 'fold', 'enrich', 'tabicl', 'ncol', 'auc', 'aupr'])
    aucs, auprs = [], []
    for f in folds:
        t0 = time.time()
        tr, ytr, te, yte = get_split(a.dataset, a.split, f, a.seed)
        Xtr, Xte, ncol = crossfit(enr, tr, ytr, te, a.seed * 100 + f)
        if a.tabicl:
            from tabicl import TabICLClassifier
            CHECKPOINT = '/data1/licheng/incrd-hg/.external/models/tabicl/tabicl-classifier-v2-20260212.ckpt'
            m = TabICLClassifier(n_estimators=32, random_state=20260914, model_path=CHECKPOINT,
                                 checkpoint_version='tabicl-classifier-v2-20260212.ckpt',
                                 allow_auto_download=False, device='cuda:0', offload_mode='auto',
                                 n_jobs=4, softmax_temperature=0.9, average_logits=True)
            m.fit(Xtr, ytr.astype(int)); s = m.predict_proba(Xte)[:, 1]
        else:
            m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=20260914)
            m.fit(Xtr, ytr); s = m.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(yte, s)); auprs.append(average_precision_score(yte, s))
        w.writerow([a.dataset, a.split, a.seed, f, a.enrich, a.tabicl, ncol,
                    f'{aucs[-1]:.6f}', f'{auprs[-1]:.6f}']); fh.flush()
        print(f'  fold {f}: AUC {aucs[-1]:.4f} AUPR {auprs[-1]:.4f} ({time.time()-t0:.0f}s)', flush=True)
    print(f'{a.dataset} {a.split} seed{a.seed} enrich={a.enrich} tabicl={a.tabicl} ncol={ncol}: '
          f'AUC {np.mean(aucs):.4f} AUPR {np.mean(auprs):.4f}', flush=True)
    fh.close()


if __name__ == '__main__':
    main()
