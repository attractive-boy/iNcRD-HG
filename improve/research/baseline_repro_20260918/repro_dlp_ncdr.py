"""Port DLP (IEEE/ACM TCBB 2026) from MATLAB to Python and run on ncDR.

Pipeline (core/*.m): Network Enhancement on the RNA similarity, KNN pre-fill,
least-squares regression (LSR) completion on both blocks, average.
"""
import os
import sys

import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

EPS = np.finfo(float).eps


def ne_dn(w, kind="ave"):
    w = w * len(w)
    if kind == "ave":
        d = 1.0 / (np.abs(w).sum(1) + EPS)
        return d[:, None] * w
    d = 1.0 / np.sqrt(np.abs(w).sum(1) + EPS)
    return (d[:, None] * w) * d[None, :]


def transition_fields(W):
    zero = np.where(W.sum(1) == 0)[0]
    W = W * len(W)
    W = ne_dn(W, "ave")
    w = np.sqrt(np.abs(W).sum(0) + EPS)
    W = W / w[None, :]
    W = W @ W.T
    W[zero, :] = 0; W[:, zero] = 0
    return W


def dominateset(aff, k):
    n = aff.shape[0]
    k = min(k, n - 1)
    idx = np.argsort(-aff, axis=1)[:, :k]
    P = np.zeros_like(aff)
    rows = np.repeat(np.arange(n), k)
    P[rows, idx.reshape(-1)] = aff[rows, idx.reshape(-1)]
    return (P + P.T) / 2


def network_enhancement(W_in, order, K, alpha):
    W_in = np.asarray(W_in, dtype=float)
    W_in1 = W_in * (1 - np.eye(len(W_in)))
    zero = np.where(np.abs(W_in1).sum(1) > 0)[0]
    W0 = W_in[np.ix_(zero, zero)]
    W = ne_dn(W0, "ave"); W = (W + W.T) / 2
    DD = np.abs(W0).sum(0)
    if len(np.unique(W)) == 2:
        P = W
    else:
        P = dominateset(np.abs(W), min(K, len(W) - 1)) * np.sign(W)
    P = P + (np.eye(len(P)) + np.diag(np.abs(P).sum(0)))
    P = transition_fields(P)
    lam, U = np.linalg.eig(P)          # numpy returns (eigenvalues, eigenvectors)
    dc = np.real(lam - EPS).astype(complex)
    dc = (1 - alpha) * dc / (1 - alpha * dc ** order)
    W = np.real(U @ np.diag(dc) @ np.linalg.inv(U))
    W = (W * (1 - np.eye(len(W)))) / (1 - np.diag(W))[:, None]
    W = np.diag(DD) @ W
    W[W < 0] = 0
    W = (W + W.T) / 2
    out = np.zeros_like(W_in)
    out[np.ix_(zero, zero)] = W
    return out


def lsr(R, alpha):
    RtR = R.T @ R
    W = np.linalg.solve((1.0 / alpha) * np.eye(RtR.shape[0]) + RtR, RtR)
    return R @ W


def knn_fill_rows(P, S, k):
    """Fill zero rows of P with similarity-weighted rows of the k nearest items."""
    P = P.copy()
    zr = np.where(P.sum(1) == 0)[0]
    for i in zr:
        nn = np.argsort(-S[i])[:k]
        P[i] = S[i, nn][None, :] @ P[nn] / (S[i, nn].sum() + EPS)
    return P


def dlp(Wdr, Wdd, Wrr, a1, a2, K):
    dn, dr = Wdr.shape
    P = Wdr.copy()
    if np.any(P.sum(1) == 0):
        P = knn_fill_rows(P, Wdd, K) + P
    T1 = np.vstack([Wrr, P])
    W1 = lsr(T1, a1)
    M1 = W1[-dn:, :dr]
    P2 = Wdr.copy()
    if np.any(P2.sum(0) == 0):
        P2 = knn_fill_rows(P2.T, Wrr, K).T + P2
    T2 = np.hstack([P2, Wdd])
    W2 = lsr(T2.T, a2).T
    M2 = W2[:dn, :dr]
    return (M1 + M2) / 2


def gip(mat):
    r = (mat ** 2).sum() / mat.shape[0]
    r = r if r > 0 else 1.0
    d = (mat ** 2).sum(1)
    S = np.exp(-(d[:, None] + d[None, :] - 2 * mat @ mat.T) / r)
    np.fill_diagonal(S, 1.0)
    return S


def eval_scores(M, pos, neg):
    pp = np.array([M[d, m] for d, m in pos])
    pn = np.array([M[d, m] for d, m in neg])
    y = np.r_[np.ones(len(pp)), np.zeros(len(pn))]
    s = np.r_[pp, pn]
    return roc_auc_score(y, s), average_precision_score(y, s)


def run_ncdr(seed):
    pairs = common.load_ncdr()[0]
    msg, tpos = common.build_split(pairs, seed)
    P = common.ncdr_pos_set(pairs)
    tneg = common.sample_neg(P, tpos.shape[1], seed + 7)
    # drug x miRNA blocks (DLP convention)
    A = np.zeros((common.ND, common.NM))
    A[msg[1], msg[0]] = 1.0
    SM = network_enhancement(gip(A.T), 0.5, 15, 0.1)   # miRNA sim
    SD = gip(A)                                        # drug sim
    M = dlp(A, SD, SM, 0.001, 0.1, 30)
    pp = np.array([M[d, m] for d, m in zip(tpos[1], tpos[0])])
    pn = np.array([M[d, m] for d, m in zip(tneg[1], tneg[0])])
    return common.full_metrics(pp, pn)


def main():
    d = sio.loadmat(os.path.join(HERE, "repos", "DLP", "data", "dataset.mat"))
    A, SM, SD = d["A"].astype(float), d["SM"], d["SD"]
    SM = network_enhancement(SM, 0.5, 15, 0.1)
    M = dlp(A, SD, SM, 0.001, 0.1, 30)
    print("DLP native dataset sanity:", M.shape, "range", float(M.min()), float(M.max()),
          "no-nan", bool(np.all(np.isfinite(M))))
    aucs, auprs = [], []
    for s in common.SEEDS:
        m = run_ncdr(s)
        print(f"ncDR seed {s}: AUC {m['AUC']:.4f} AUPR {m['AUPR']:.4f} ACC {m['ACC']:.4f} F1 {m['F1']:.4f} MCC {m['MCC']:.4f}")
        aucs.append(m["AUC"]); auprs.append(m["AUPR"])
    print(f"DLP (ncDR same-protocol): AUC {np.mean(aucs):.4f}+-{np.std(aucs):.4f} "
          f"AUPR {np.mean(auprs):.4f}+-{np.std(auprs):.4f}")


if __name__ == "__main__":
    main()
