# -*- coding: utf-8 -*-
"""Which additional query-masked pair quantities carry signal beyond qis + seq relations + masked EASE?
LR probe, PLMF-MDA 5cv seed 42.  All quantities exact with the query edge removed."""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, '.')
from feat_probe import G, rowminmax, knn, stem, same, feats
from ease_probe import ease_item, ease_scores
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921')
from plmf_splits import get_split, load_dataset


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def p5_masked(A, r, d):
    Gm = A.T @ A; out = np.zeros(len(r))
    base = (A @ Gm @ Gm)[r, d]
    for k in range(len(r)):
        if A[r[k], d[k]] == 0:
            out[k] = base[k]; continue
        am = A[r[k]].copy(); e = np.zeros(A.shape[1]); e[d[k]] = 1
        G2 = Gm - np.outer(am, e) - np.outer(e, am) + np.outer(e, e)
        am2 = am - e
        out[k] = am2 @ G2 @ G2 @ e
    return out


def katz_masked(A, Sr, Sd, r, d, beta, gamma):
    nr, nd = A.shape; n = nr + nd
    B = np.zeros((n, n)); B[:nr, nr:] = A; B[nr:, :nr] = A.T; B[:nr, :nr] = gamma * Sr; B[nr:, nr:] = gamma * Sd
    rho = np.max(np.abs(np.linalg.eigvalsh(B)))
    b = beta / rho
    K = np.linalg.inv(np.eye(n) - b * B)
    dt = nr + d
    out = K[r, dt].copy()
    q = A[r, d]
    # remove edge: (I - bB') = (I - bB) + b (e_m e_d^T + e_d e_m^T) ; Woodbury with C = b [[0,1],[1,0]]
    Kmm, Kmd, Kdd = K[r, r], K[r, dt], K[dt, dt]
    for k in np.where(q > 0)[0]:
        Cinv = np.array([[0.0, 1.0], [1.0, 0.0]]) / b
        W = np.array([[Kmm[k], Kmd[k]], [Kmd[k], Kdd[k]]])
        M = Cinv + W
        # entry [m, d] of K - K W (M^-1) W^T K restricted: [K_mm K_md] M^-1 [K_md K_dd]^T
        v1 = np.array([Kmm[k], Kmd[k]]); v2 = np.array([Kmd[k], Kdd[k]])
        out[k] = Kmd[k] - v1 @ np.linalg.solve(M, v2)
    return out


def extra(A, xr, xd, Sr, Sd, r, d):
    q = A[r, d]
    ur, ud = unit(xr), unit(xd)
    # masked profile cosines
    prof_d = A @ ud - q[:, None] * ud[d] if False else None
    Pd = A[r] @ ud - q[:, None] * ud[d]                 # sum of other drugs' embeddings
    Pr = A[:, d].T @ ur - q[:, None] * ur[r]
    cos_d = (unit(Pd) * ud[d]).sum(1); cos_r = (unit(Pr) * ur[r]).sum(1)
    Cr = ur @ ur.T; np.fill_diagonal(Cr, 0); Cd = ud @ ud.T; np.fill_diagonal(Cd, 0)
    dens_r = (Cr @ A)[r, d]; dens_d = (A @ Cd)[r, d]
    degd = A.sum(0); degr = A.sum(1)
    nbr_deg_r = (A[r] @ degd - q * degd[d]) / np.maximum(degr[r] - q, 1)      # mean degree of m's other drugs
    nbr_deg_d = (A[:, d].T @ degr - q * degr[r]) / np.maximum(degd[d] - q, 1)
    # note: other drugs' degrees include the (m, d') edges (not the query edge) -> exclusive
    return dict(cos=np.stack([cos_d, cos_r], 1), dens=np.stack([dens_r, dens_d], 1), nbrdeg=np.stack([nbr_deg_r, nbr_deg_d], 1))


def lr(Xtr, ytr, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    m = LogisticRegression(C=1.0, max_iter=5000).fit((Xtr - mu) / sd, ytr)
    s = m.predict_proba((Xte - mu) / sd)[:, 1]
    return roc_auc_score(yte, s), average_precision_score(yte, s)


def main(ds):
    g = np.load(G.format(ds), allow_pickle=True)
    mir, dru, pos, neg = load_dataset(ds)
    names = mir['miRNA_Name'].astype(str).values; seqs = mir['miRNA_Sequence'].astype(str).str.upper().str.replace('T', 'U').tolist()
    xr = rowminmax(g['mirna_kmer'].astype(float)); xd = rowminmax(g['drug_x'].astype(float))
    Sr, Sd = knn(xr, 10), knn(xd, 10)
    rels = [same([s[1:8] for s in seqs]), same([stem(n) for n in names])]
    rows = []
    for f in range(5):
        tr, ytr, te, yte = get_split(ds, '5cv', f, 42)
        A = np.zeros((len(names), len(dru))); A[tr[ytr == 1, 0], tr[ytr == 1, 1]] = 1
        blocks_tr, blocks_te = {}, {}
        blocks_tr['qis'] = feats(A, Sr, Sd, rels, tr[:, 0], tr[:, 1], True); blocks_te['qis'] = feats(A, Sr, Sd, rels, te[:, 0], te[:, 1], True)
        Pd = ease_item(A, 10.0); Pr = ease_item(A.T, 10.0)
        blocks_tr['ease'] = np.stack([ease_scores(A, Pd, tr[:, 0], tr[:, 1], True), ease_scores(A.T, Pr, tr[:, 1], tr[:, 0], True)], 1)
        blocks_te['ease'] = np.stack([ease_scores(A, Pd, te[:, 0], te[:, 1], True), ease_scores(A.T, Pr, te[:, 1], te[:, 0], True)], 1)
        blocks_tr['p5'] = np.log1p(p5_masked(A, tr[:, 0], tr[:, 1]))[:, None]; blocks_te['p5'] = np.log1p(p5_masked(A, te[:, 0], te[:, 1]))[:, None]
        kt, ke = [], []
        for beta, gam in ((0.5, 0.0), (0.9, 0.0), (0.5, 0.3), (0.9, 0.3)):
            kt.append(katz_masked(A, Sr, Sd, tr[:, 0], tr[:, 1], beta, gam)); ke.append(katz_masked(A, Sr, Sd, te[:, 0], te[:, 1], beta, gam))
        tf = lambda x: np.sign(x) * np.log1p(np.abs(x) * 100)
        blocks_tr['katz'] = tf(np.stack(kt, 1)); blocks_te['katz'] = tf(np.stack(ke, 1))
        et = extra(A, xr, xd, Sr, Sd, tr[:, 0], tr[:, 1]); ee = extra(A, xr, xd, Sr, Sd, te[:, 0], te[:, 1])
        for k in et:
            blocks_tr[k] = np.log1p(np.abs(et[k])) * np.sign(et[k]); blocks_te[k] = np.log1p(np.abs(ee[k])) * np.sign(ee[k])
        base = ['qis', 'ease']
        combos = {'base': base, '+p5': base + ['p5'], '+katz': base + ['katz'], '+cos': base + ['cos'], '+dens': base + ['dens'],
                  '+nbrdeg': base + ['nbrdeg'], '+all': base + ['p5', 'katz', 'cos', 'dens', 'nbrdeg'], 'katz_only': ['katz']}
        for name, bl in combos.items():
            au, ap_ = lr(np.hstack([blocks_tr[b] for b in bl]), ytr, np.hstack([blocks_te[b] for b in bl]), yte)
            rows.append(dict(ds=ds, fold=f, combo=name, auc=au, aupr=ap_))
    d = pd.DataFrame(rows)
    print(d.groupby(['ds', 'combo'])[['auc', 'aupr']].mean().round(4))


if __name__ == '__main__':
    for ds in ('MDR', 'MDS'):
        main(ds)
