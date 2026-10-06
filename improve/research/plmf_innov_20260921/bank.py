# -*- coding: utf-8 -*-
"""Feature banks for the PLMF-MDA innovation study.

All features are strictly label-free at query time: pair features for a query
use only a *support* set of positive pairs (cross-fitted for the training split,
full training-positive support for the evaluation split), exactly as
dlstmda_strongest_20260920/features.py::crossfit_features.

Families
  ctx      : the 33 graph-context columns (GraphPairEncoder)            [base]
  diff     : personalized PageRank / Katz / path-7 on the support graph  [enrich_probe]
  scf      : spectral collaborative filtering -- elementwise products of the
             top-K singular vectors of the degree-normalised support matrix,
             plus per-side collaborative embeddings (novel)
  align    : content<->collaborative alignment columns (novel)
"""
import hashlib
import numpy as np
import pandas as pd

from features import GraphPairEncoder  # noqa: E402  (dlstmda_strongest_20260920)

DIFF = ((0.3, 0.01), (0.6, 0.05))
KMAX = 12


def diffusion(A, alpha, beta):
    nr, nd = A.shape
    dsum = np.maximum(A.sum(1, keepdims=True), 1)
    ddsum = np.maximum(A.sum(0, keepdims=True), 1)
    An = A / dsum
    And = (A / ddsum).T
    G = And @ An
    Gs = An @ And
    ppr = np.zeros((nr, nd)); katz = np.zeros((nr, nd))
    p = An.copy()
    for k in range(KMAX):
        ppr += alpha * (1 - alpha) ** (2 * k + 1) * p
        katz += beta ** (2 * k + 1) * p
        p = p @ G
    ppr_d = np.zeros((nd, nr)); katz_d = np.zeros((nd, nr))
    q = And.copy()
    for k in range(KMAX):
        ppr_d += alpha * (1 - alpha) ** (2 * k + 1) * q
        katz_d += beta ** (2 * k + 1) * q
        q = q @ Gs
    return dict(ppr_rd=ppr, katz_rd=katz, ppr_dr=ppr_d.T, katz_dr=katz_d.T,
                path7=(A @ np.linalg.matrix_power(G, 3)))


def content_diffusion(sr, sd, A, alpha=0.15, gamma=0.5, steps=10):
    """Personalized PageRank from every miRNA on the content-augmented
    heterogeneous graph  [[gamma*sr, A],[A^T, gamma*sd]].

    A plain bipartite walk only reaches drug j from miRNA i through shared
    training positives.  Allowing miRNA-miRNA (content) and drug-drug (content)
    hops lets the walk travel i -> i' (similar miRNA) -> j, and i -> j' (similar
    drug), which a bipartite-only diffusion cannot express.  Returns R[nr,nd],
    the PPR mass at each drug for a restart at each miRNA.
    """
    nr, nd = A.shape
    A = np.asarray(A, dtype=np.float64); sr = np.asarray(sr, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    Prr = sr / np.maximum(sr.sum(1, keepdims=True), 1e-8)
    Pdd = sd / np.maximum(sd.sum(0, keepdims=True), 1e-8)
    Prd = A / np.maximum(A.sum(1, keepdims=True), 1e-8)
    Pdr = A.T / np.maximum(A.sum(0)[:, None], 1e-8)
    X = np.eye(nr)                      # distribution over miRNA nodes
    Y = np.zeros((nr, nd))              # distribution over drug nodes
    R = np.zeros((nr, nd))
    for _ in range(steps):
        R += alpha * Y
        Xn = (1 - alpha) * (X @ Prr + gamma * Y @ Pdr) + alpha * np.eye(nr)
        Yn = (1 - alpha) * (gamma * X @ Prd + Y @ Pdd)
        rs = np.maximum(Xn.sum(1, keepdims=True) + Yn.sum(1, keepdims=True), 1e-8)
        X = Xn / rs; Y = Yn / rs
    R += alpha * Y
    return R


def spec_cf(A, K=16):
    """Top-K spectral collaborative-filtering quantities of the support matrix."""
    nr, nd = A.shape
    dr = np.maximum(A.sum(1), 1e-8)
    dd = np.maximum(A.sum(0), 1e-8)
    M = (A / np.sqrt(dr)[:, None]) / np.sqrt(dd)[None, :]
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    V = Vt.T
    K = min(K, len(s))
    return U[:, :K], s[:K], V[:, :K]


class Bank:
    def __init__(self, xr, xd, sr, sd):
        self.enc = GraphPairEncoder(xr, xd, sr, sd)
        self.nr, self.nd = xr.shape[0], xd.shape[0]
        self.sr = sr.astype(np.float64)
        self.sd = sd.astype(np.float64)
        self.ur = xr / np.maximum(np.linalg.norm(xr, axis=1, keepdims=True), 1e-8)
        self.ud = xd / np.maximum(np.linalg.norm(xd, axis=1, keepdims=True), 1e-8)
        self.cache = {}

    # ---- base + diffusion ------------------------------------------------
    def _base(self, sup, q):
        return self.enc.transform(np.asarray(q, dtype=np.int64).T,
                                  np.asarray(sup, dtype=np.int64).T)

    def _global(self, sup):
        sup = np.asarray(sup, dtype=np.int64)
        rr, cc = sup[:, 0], sup[:, 1]
        k = hashlib.md5(np.sort(rr * 100000 + cc).tobytes()).hexdigest() + 'g'
        if k in self.cache:
            return self.cache[k]
        A = np.zeros((self.nr, self.nd)); A[rr, cc] = 1.0
        out = {}
        for al, be in DIFF:
            for key, D in diffusion(A, al, be).items():
                out[f'{key}_{int(al*100)}_{int(be*1000)}'] = D
        out['srA'] = self.sr @ A
        out['ASd'] = A @ self.sd
        out['srASd'] = self.sr @ A @ self.sd
        self.cache[k] = out
        return out

    def _diff_cols(self, sup, q):
        gm = self._global(sup)
        r, d = q[:, 0], q[:, 1]
        cols = {}
        for k, D in gm.items():
            v = D[r, d]
            cols[f'd_{k}'] = v
            cols[f'd_{k}_rz'] = (v - D[r].mean(1)) / (D[r].std(1) + 1e-8)
            cols[f'd_{k}_rpc'] = (D[r] < v[:, None]).mean(1)
            cols[f'd_{k}_dz'] = (v - D[:, d].mean(0)) / (D[:, d].std(0) + 1e-8)
        return cols

    def _cdiff_cols(self, sup, q):
        sup = np.asarray(sup, dtype=np.int64)
        rr, cc = sup[:, 0], sup[:, 1]
        key = hashlib.md5(np.sort(rr * 100000 + cc).tobytes()).hexdigest() + 'cd'
        if key not in self.cache:
            A = np.zeros((self.nr, self.nd)); A[rr, cc] = 1.0
            self.cache[key] = {
                f'cdiff_{int(al*100)}_g{int(g*10)}': content_diffusion(self.sr, self.sd, A, al, g)
                for al in (0.1, 0.25) for g in (0.3, 1.0)}
        r, d = q[:, 0], q[:, 1]
        cols = {}
        for k, D in self.cache[key].items():
            v = D[r, d]
            cols[k] = v
            cols[k + '_rz'] = (v - D[r].mean(1)) / (D[r].std(1) + 1e-8)
            cols[k + '_rpc'] = (D[r] < v[:, None]).mean(1)
            cols[k + '_dz'] = (v - D[:, d].mean(0)) / (D[:, d].std(0) + 1e-8)
        return cols

    # ---- spectral CF -----------------------------------------------------
    def _scf(self, sup, q, K):
        sup = np.asarray(sup, dtype=np.int64)
        rr, cc = sup[:, 0], sup[:, 1]
        key = hashlib.md5(np.sort(rr * 100000 + cc).tobytes()).hexdigest() + f's{K}'
        if key in self.cache:
            U, s, V = self.cache[key]
        else:
            A = np.zeros((self.nr, self.nd)); A[rr, cc] = 1.0
            U, s, V = spec_cf(A, K)
            self.cache[key] = (U, s, V)
        r, d = q[:, 0], q[:, 1]
        k = U.shape[1]
        P = U[r] * V[d]                       # elementwise products, k cols
        cols = {f'scf_prod_{i}': P[:, i] for i in range(k)}
        cols['scf_dot'] = (U[r] * s * V[d]).sum(1)
        cols['scf_rnorm'] = np.linalg.norm(U[r], axis=1)
        cols['scf_dnorm'] = np.linalg.norm(V[d], axis=1)
        return cols, U, s, V

    def _align(self, sup, q, U, V):
        """Content-neighbourhood collaborative alignment for a query pair."""
        r, d = q[:, 0], q[:, 1]
        wr = self.sr / np.maximum(self.sr.sum(1, keepdims=True), 1e-8)
        wd = self.sd / np.maximum(self.sd.sum(0, keepdims=True), 1e-8)
        U_nb = wr @ U          # miRNA content-neighbour avg collaborative embedding
        V_nb = wd @ V          # drug  content-neighbour avg collaborative embedding
        return dict(
            align_r_nb_d=(U_nb[r] * V[d]).sum(1),
            align_r_d_nb=(U[r] * V_nb[d]).sum(1),
            align_nb_nb=(U_nb[r] * V_nb[d]).sum(1),
            align_r_d=(U[r] * V[d]).sum(1),
        )

    def transform(self, sup, q, families):
        cols = {}
        if 'ctx' in families:
            base = self._base(sup, q)
            for c in base.columns:
                cols[c] = base[c].values
        if 'diff' in families:
            cols.update(self._diff_cols(sup, q))
        if 'cdiff' in families:
            cols.update(self._cdiff_cols(sup, q))
        if families & {'scf', 'align'}:
            K = 16
            sc, U, s, V = self._scf(sup, q, K)
            if 'scf' in families:
                cols.update(sc)
            if 'align' in families:
                cols.update(self._align(sup, q, U, V))
        cols = {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}
        return pd.DataFrame(cols)
