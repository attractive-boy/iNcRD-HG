"""Reproduce GCMDR (Bioinformatics 2019, local PyG port) on the ncDR split."""
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
LATENT = 25
LR = 1e-3
EPOCH = 200
WD = 5e-4
DROPOUT = 0.1

kmer = pd.read_csv(f"{DATA}/data/miRNA_kmer.csv", header=None).values.astype(np.float32)
gin = pd.read_csv(f"{DATA}/data/drug_GIN_64.csv", header=None).values.astype(np.float32)


def seed_it(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class GCMDR(nn.Module):
    def __init__(self, N, in_dim, latent):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(in_dim, latent)); nn.init.xavier_uniform_(self.W1)
        self.W2 = nn.Parameter(torch.zeros(N, latent))
        self.W3 = nn.Parameter(torch.empty(latent, latent)); nn.init.xavier_uniform_(self.W3)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, adj, X):
        H = F.relu(torch.sparse.mm(adj, X) @ self.W1) + self.W2
        return self.drop(H)


def run_seed(seed):
    pairs = common.load_ncdr()[0]
    msg, tpos = common.build_split(pairs, seed)
    P = common.ncdr_pos_set(pairs)
    trneg = common.sample_neg(P, msg.shape[1], seed + 11)
    tneg = common.sample_neg(P, tpos.shape[1], seed + 7)
    nc, nd = common.NM, common.ND
    N = nc + nd
    X = np.zeros((N, kmer.shape[1] + gin.shape[1]), dtype=np.float32)
    X[:nc, :kmer.shape[1]] = kmer
    X[nc:, kmer.shape[1]:] = gin
    s = X.sum(1, keepdims=True); s[s == 0] = 1; X = X / s
    R = sp.coo_matrix((np.ones(msg.shape[1]), (msg[0], msg[1])), shape=(nc, nd)).tocsr()
    A = sp.vstack([sp.hstack([sp.csr_matrix((nc, nc)), R]),
                   sp.hstack([R.T, sp.csr_matrix((nd, nd))])]).astype(float)
    A = A + sp.eye(N)
    deg = np.array(A.sum(1)).flatten()
    dinv = np.power(deg, -0.5); dinv[np.isinf(dinv)] = 0
    D = sp.diags(dinv)
    nA = (D @ A @ D).tocoo()
    idx = torch.tensor(np.vstack([nA.row, nA.col]), dtype=torch.long)
    val = torch.tensor(nA.data, dtype=torch.float32)
    adjT = torch.sparse_coo_tensor(idx, val, (N, N)).coalesce().to(DEV)
    Xt = torch.tensor(X, dtype=torch.float32, device=DEV)

    tr = np.vstack([msg.T, trneg.T])
    tr_lab = np.r_[np.ones(msg.shape[1]), np.zeros(trneg.shape[1])]
    tr_idx = torch.tensor(tr.T, dtype=torch.long, device=DEV)
    tr_lab = torch.tensor(tr_lab, dtype=torch.float32, device=DEV)

    seed_it(seed)
    model = GCMDR(N, X.shape[1], LATENT).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    for ep in range(EPOCH):
        model.train(); opt.zero_grad()
        H = model(adjT, Xt)
        M = H[:nc] @ model.W3 @ H[nc:].t()
        loss = F.binary_cross_entropy_with_logits(M[tr_idx[0], tr_idx[1]], tr_lab)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        H = model(adjT, Xt)
        M = H[:nc] @ model.W3 @ H[nc:].t()
        pp = M[torch.tensor(tpos[0], device=DEV), torch.tensor(tpos[1], device=DEV)].cpu().numpy()
        pn = M[torch.tensor(tneg[0], device=DEV), torch.tensor(tneg[1], device=DEV)].cpu().numpy()
    return common.full_metrics(pp, pn)


def main():
    ms = []
    for s in common.SEEDS:
        m = run_seed(s)
        ms.append(m)
        print(f"seed {s}: AUC {m['AUC']:.4f} AUPR {m['AUPR']:.4f} ACC {m['ACC']:.4f} F1 {m['F1']:.4f} MCC {m['MCC']:.4f}")
    print("\nGCMDR (ncDR same-protocol) means:")
    for k in ("AUC", "AUPR", "ACC", "Precision", "Recall", "Specificity", "F1", "MCC"):
        v = np.array([m[k] for m in ms]); print(f"  {k:12s} {v.mean():.4f} +- {v.std():.4f}")


if __name__ == "__main__":
    main()
