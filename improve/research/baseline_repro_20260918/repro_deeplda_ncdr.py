"""Reproduce DeepLDA (Front Microbiol 2023, local port) on the ncDR split."""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

DATA = "/tmp/opencode/comp/GAMMDR"
DEV = "cuda:0"
LR, EPOCH, HID, DROP, WD = 5e-4, 400, 256, 0.3, 1e-4

kmer = pd.read_csv(f"{DATA}/data/miRNA_kmer.csv", header=None).values.astype(np.float32)
gin = pd.read_csv(f"{DATA}/data/drug_GIN_64.csv", header=None).values.astype(np.float32)


def gip(mat):
    r = (mat ** 2).sum() / mat.shape[0]
    r = r if r > 0 else 1.0
    d = (mat ** 2).sum(1)
    S = np.exp(-(d[:, None] + d[None, :] - 2 * mat @ mat.T) / r)
    np.fill_diagonal(S, 1.0)
    return S


def knn_edges(S, k):
    n = S.shape[0]; k = min(k, n - 1)
    idx = np.argpartition(-S, kth=k, axis=1)[:, :k]
    r = np.repeat(np.arange(n), k)
    return torch.tensor(np.stack([r, idx.reshape(-1)]), dtype=torch.long)


class DeepLDA(nn.Module):
    def __init__(self, in_c, hid, num_rna, drop=0.3):
        super().__init__()
        self.gcn1 = pyg_nn.GCNConv(in_c, hid)
        self.gat1 = pyg_nn.GATConv(in_c, hid, heads=2)
        self.lin = nn.Linear(hid + hid * 2, hid)
        self.drop = nn.Dropout(drop)
        self.num_rna = num_rna

    def forward(self, x, ei):
        a = self.gcn1(x, ei); b = self.gat1(x, ei)
        x = self.lin(self.drop(torch.cat([a, b], 1)))
        return torch.matmul(x[:self.num_rna], x[self.num_rna:].T)


def run_seed(seed):
    pairs = common.load_ncdr()[0]
    msg, tpos = common.build_split(pairs, seed)
    P = common.ncdr_pos_set(pairs)
    trneg = common.sample_neg(P, msg.shape[1], seed + 11)
    tneg = common.sample_neg(P, tpos.shape[1], seed + 7)
    nm, nd = common.NM, common.ND
    R = np.zeros((nm, nd), dtype=np.float32); R[msg[0], msg[1]] = 1
    er = knn_edges(gip(R), 10); ed = knn_edges(gip(R.T), 10)
    ed = ed + nm
    e_tr = torch.tensor(msg, dtype=torch.long)
    sim = torch.cat([er, er.flip(0), ed, ed.flip(0),
                     e_tr, e_tr.flip(0), e_tr.flip(0)], dim=1)
    # build undirected train/test graph (association edges both directions)
    def build_ei(pos):
        e = torch.tensor(pos, dtype=torch.long)
        e2 = torch.stack([e[0], e[1] + nm])
        return torch.cat([sim, e2, e2.flip(0)], dim=1)
    X = torch.zeros((nm + nd, kmer.shape[1] + gin.shape[1]))
    X[:nm, :kmer.shape[1]] = torch.tensor(kmer)
    X[nm:, kmer.shape[1]:] = torch.tensor(gin)
    s = X.sum(1, keepdims=True); s[s == 0] = 1; X = (X / s).to(DEV)
    d_tr = Data(x=X, edge_index=build_ei(msg).to(DEV))
    d_te = Data(x=X, edge_index=build_ei(tpos).to(DEV))
    tr_idx = torch.tensor(np.vstack([msg[0], msg[1]]), dtype=torch.long, device=DEV)
    tr_lab = torch.cat([torch.ones(msg.shape[1]), torch.zeros(trneg.shape[1])]).float().to(DEV)
    tr_idx = torch.cat([tr_idx, torch.tensor(np.vstack([trneg[0], trneg[1]]), dtype=torch.long, device=DEV)], 1)
    torch.manual_seed(seed)
    model = DeepLDA(X.shape[1], HID, nm, DROP).to(DEV)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    for ep in range(EPOCH):
        model.train(); opt.zero_grad()
        o = model(d_tr.x, d_tr.edge_index)[tr_idx[0], tr_idx[1]]
        nn.BCEWithLogitsLoss()(o, tr_lab).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        M = model(d_te.x, d_te.edge_index)
        pp = M[torch.tensor(tpos[0], device=DEV), torch.tensor(tpos[1], device=DEV)].cpu().numpy()
        pn = M[torch.tensor(tneg[0], device=DEV), torch.tensor(tneg[1], device=DEV)].cpu().numpy()
    return common.full_metrics(pp, pn)


def main():
    ms = []
    for s in common.SEEDS:
        m = run_seed(s); ms.append(m)
        print(f"seed {s}: AUC {m['AUC']:.4f} AUPR {m['AUPR']:.4f} ACC {m['ACC']:.4f} F1 {m['F1']:.4f} MCC {m['MCC']:.4f}")
    print("\nDeepLDA (ncDR same-protocol) means:")
    for k in ("AUC", "AUPR", "ACC", "Precision", "Recall", "Specificity", "F1", "MCC"):
        v = np.array([m[k] for m in ms]); print(f"  {k:12s} {v.mean():.4f} +- {v.std():.4f}")


if __name__ == "__main__":
    main()
