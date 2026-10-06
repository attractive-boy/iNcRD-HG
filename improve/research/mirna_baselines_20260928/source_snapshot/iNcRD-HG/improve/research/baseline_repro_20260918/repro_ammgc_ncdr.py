"""Reproduce AMMGC (Front Pharmacol 2022) on the ncDR split.

Official code is TF1 with hardcoded 106x754 dims and 4 multimodal feature views.
We faithfully reimplement its architecture in PyTorch: block adjacency
[drug | miRNA], 4 GCN encoders, attention-based view fusion, inner-product decoder.
The 4 views are built from the ncDR node features (kmer for miRNA, GIN for drug)
and training-set GIP self-similarities.
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

DATA = "/tmp/opencode/comp/GAMMDR"
DEV = "cuda:0"
HID, OUT = 128, 256
DROPOUT = 0.1
LR = 1e-2
EPOCH = 200

kmer = pd.read_csv(f"{DATA}/data/miRNA_kmer.csv", header=None).values.astype(np.float32)
gin = pd.read_csv(f"{DATA}/data/drug_GIN_64.csv", header=None).values.astype(np.float32)


def gip(mat):
    r = (mat ** 2).sum() / mat.shape[0]
    r = r if r > 0 else 1.0
    d = (mat ** 2).sum(1)
    S = np.exp(-(d[:, None] + d[None, :] - 2 * mat @ mat.T) / r)
    np.fill_diagonal(S, 1.0)
    return S.astype(np.float32)


class GCN(nn.Module):
    def __init__(self, in_dim, hid, out):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hid)
        self.l2 = nn.Linear(hid, out)
        self.d = nn.Dropout(DROPOUT)

    def forward(self, A, X):
        X = self.d(X)
        X = F.relu(self.l1(torch.sparse.mm(A, X)))
        X = self.d(X)
        X = self.l2(torch.sparse.mm(A, X))
        return X


class AMMGC(nn.Module):
    def __init__(self, dims, out):
        super().__init__()
        self.enc = nn.ModuleList([GCN(d, HID, out) for d in dims])
        self.att = nn.Parameter(torch.zeros(len(dims), out))

    def forward(self, A, Xs):
        embs = [e(A, X) for e, X in zip(self.enc, Xs)]
        stack = torch.stack(embs, 0)                      # K x N x out
        logits = (stack * self.att[:, None, :]).sum(-1)   # K x N
        alpha = torch.softmax(logits, 0)                  # K x N
        return (stack * alpha[:, :, None]).sum(0)         # N x out


def normalize_adj(R, nd, nm):
    N = nd + nm
    A = sp.vstack([sp.hstack([sp.csr_matrix((nd, nd)), R]),
                   sp.hstack([R.T, sp.csr_matrix((nm, nm))])]).astype(float)
    A = A + sp.eye(N)
    deg = np.array(A.sum(1)).flatten()
    dinv = np.power(deg, -0.5); dinv[np.isinf(dinv)] = 0
    D = sp.diags(dinv)
    nA = (D @ A @ D).tocoo()
    idx = torch.tensor(np.vstack([nA.row, nA.col]), dtype=torch.long)
    val = torch.tensor(nA.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (N, N)).coalesce().to(DEV)


def build_views(m_sim, d_sim, nd, nm):
    views, dims = [], []
    combos = [(gin, kmer), (gin, m_sim), (d_sim, kmer), (d_sim, m_sim)]
    width = max(a.shape[1] + b.shape[1] for a, b in combos)
    for a, b in combos:
        X = np.zeros((nd + nm, width), dtype=np.float32)
        X[:nd, :a.shape[1]] = a
        X[nd:, a.shape[1]:a.shape[1] + b.shape[1]] = b
        s = X.sum(1, keepdims=True); s[s == 0] = 1
        views.append(torch.tensor(X / s, device=DEV))
        dims.append(width)
    return views, dims


def run_seed(seed):
    pairs = common.load_ncdr()[0]
    msg, tpos = common.build_split(pairs, seed)
    P = common.ncdr_pos_set(pairs)
    trneg = common.sample_neg(P, msg.shape[1], seed + 11)
    tneg = common.sample_neg(P, tpos.shape[1], seed + 7)
    nd, nm = common.ND, common.NM
    R = sp.coo_matrix((np.ones(msg.shape[1]), (msg[1], msg[0])), shape=(nd, nm)).tocsr()
    A = normalize_adj(R, nd, nm)
    Rf = np.zeros((nm, nd), dtype=np.float32); Rf[msg[0], msg[1]] = 1
    m_sim = gip(Rf); d_sim = gip(Rf.T)
    views, dims = build_views(m_sim, d_sim, nd, nm)

    seed_it(seed)
    model = AMMGC(dims, OUT).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    tr = np.vstack([msg.T, trneg.T])
    tr_lab = torch.tensor(np.r_[np.ones(msg.shape[1]), np.zeros(trneg.shape[1])], dtype=torch.float32, device=DEV)
    tr_d = torch.tensor(tr[:, 1], dtype=torch.long, device=DEV)
    tr_m = torch.tensor(tr[:, 0], dtype=torch.long, device=DEV)
    for ep in range(EPOCH):
        model.train(); opt.zero_grad()
        Z = model(A, views)
        logits = (Z[:nd][tr_d] * Z[nd:][tr_m]).sum(-1)
        loss = F.binary_cross_entropy_with_logits(logits, tr_lab)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        Z = model(A, views)
        pd_ = torch.tensor(tpos[1], dtype=torch.long, device=DEV)
        pm_ = torch.tensor(tpos[0], dtype=torch.long, device=DEV)
        pp = (Z[:nd][pd_] * Z[nd:][pm_]).sum(-1).cpu().numpy()
        tnd = torch.tensor(tneg[1], dtype=torch.long, device=DEV)
        tnm = torch.tensor(tneg[0], dtype=torch.long, device=DEV)
        pn = (Z[:nd][tnd] * Z[nd:][tnm]).sum(-1).cpu().numpy()
    return common.full_metrics(pp, pn)


def seed_it(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    ms = []
    for s in common.SEEDS:
        m = run_seed(s); ms.append(m)
        print(f"seed {s}: AUC {m['AUC']:.4f} AUPR {m['AUPR']:.4f} ACC {m['ACC']:.4f} F1 {m['F1']:.4f} MCC {m['MCC']:.4f}")
    print("\nAMMGC (ncDR same-protocol) means:")
    for k in ("AUC", "AUPR", "ACC", "Precision", "Recall", "Specificity", "F1", "MCC"):
        v = np.array([m[k] for m in ms]); print(f"  {k:12s} {v.mean():.4f} +- {v.std():.4f}")


if __name__ == "__main__":
    main()
