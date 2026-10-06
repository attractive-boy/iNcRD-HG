"""Offline probe on full-matrix hollow logits (valsel epoch): does a matrix-level operation help?
 (1) column/row quota calibration: per-drug (per-miRNA) offset b so that the prior-shifted probabilities of the
     drug's non-graph cells sum to the expected number of hidden positives (graph degree x hidden fraction);
 (2) low-rank smoothing of the logit matrix (top-k SVD of the full hollow logit matrix), blended with the logit.
Only training information enters (graph degrees, model scores of all cells); test labels are used only to score."""
import sys, numpy as np
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
from scipy.special import expit, logit
from scipy.optimize import brentq
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'Fd'
res = {}
for f in range(5):
    z = np.load(f'cache/dump_{tag}_{ds}_5cv_s42_f{f}.npz')
    tp = z['train_pos']; nm = int(max(z['u'][:, 0].max(), z['tr'][:, 0].max(), z['te'][:, 0].max())) + 1
    nd = int(max(z['u'][:, 1].max(), z['tr'][:, 1].max(), z['te'][:, 1].max())) + 1
    L = np.full((nm, nd), np.nan)
    for part in ('u', 'tr', 'va', 'te'):
        P = z[part]; L[P[:, 0], P[:, 1]] = z[f'logit_{part}']
    assert not np.isnan(L).any()
    A = np.zeros((nm, nd)); A[tp[:, 0], tp[:, 1]] = 1
    te, yte, lte = z['te'], z['yte'], z['logit_te']
    base = (AU(yte, lte), AP(yte, lte))
    res.setdefault('base', []).append(base)
    # hidden fraction: positives outside the graph / positives in graph.  5cv: graph = 80% of all positives -> 0.25
    hid = 0.25
    NG = A == 0
    # global prior shift c: calibrate so the total over non-graph cells = hid * |E|
    tot = hid * A.sum()
    c = brentq(lambda c: expit(L[NG] + c).sum() - tot, -30, 30)
    for side in ('col', 'row', 'both'):
        B = np.zeros_like(L)
        if side in ('col', 'both'):
            for d in range(nd):
                ng = NG[:, d]; tgt = max(hid * A[:, d].sum(), 0.25)
                b = brentq(lambda b: expit(L[ng, d] + c + b).sum() - tgt, -40, 40)
                B[:, d] += b
        if side in ('row', 'both'):
            for m in range(nm):
                ng = NG[m]; tgt = max(hid * A[m].sum(), 0.5)
                b = brentq(lambda b: expit(L[m, ng] + c + B[m, ng] + b).sum() - tgt, -40, 40)
                B[m, :] += b
        for lam in (0.25, 0.5, 1.0):
            s = lte + lam * B[te[:, 0], te[:, 1]]
            res.setdefault(f'quota_{side}_{lam}', []).append((AU(yte, s), AP(yte, s)))
    # low-rank smoothing (probability space, prior-shifted) -- rank k SVD of the full hollow score matrix
    Pm = expit(L)
    U, S, Vt = np.linalg.svd(Pm, full_matrices=False)
    for k in (5, 10, 20, 40):
        Lk = (U[:, :k] * S[:k]) @ Vt[:k]
        for lam in (0.5, 1.0, 2.0):
            s = lte + lam * logit(np.clip(Lk[te[:, 0], te[:, 1]], 1e-4, 1 - 1e-4))
            res.setdefault(f'svd{k}_{lam}', []).append((AU(yte, s), AP(yte, s)))
b = np.mean(res['base'], 0)
print(f'{ds} base {b[0]:.4f}/{b[1]:.4f}')
for k, v in res.items():
    if k == 'base': continue
    m = np.mean(v, 0); w = np.mean([x[0] > y[0] for x, y in zip(v, res['base'])])
    print(f'  {k:18s} {m[0]:.4f}/{m[1]:.4f}  d {m[0]-b[0]:+.4f}/{m[1]-b[1]:+.4f}  winAUC {w:.1f}')
