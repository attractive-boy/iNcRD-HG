"""Re-run GAM-MDR (local port, /tmp/opencode/exp_final.py) on the ncDR split and
report the full metric set. Also re-derives our GNN edge decoder for reference.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, negative_sampling

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

DATA = "/tmp/opencode/comp/GAMMDR"
kmer = pd.read_csv(f"{DATA}/data/miRNA_kmer.csv", header=None).values.astype(np.float32)
gin = pd.read_csv(f"{DATA}/data/drug_GIN_64.csv", header=None).values.astype(np.float32)
feat = torch.tensor(np.vstack([kmer, gin]))


class BE(torch.nn.Module):
    def __init__(self, ic, h, oc, L):
        super().__init__()
        self.cv = torch.nn.ModuleList([GCNConv(ic if i == 0 else h, oc if i == L - 1 else h) for i in range(L)])
        self.bn = torch.nn.ModuleList([torch.nn.BatchNorm1d(oc if i == L - 1 else h) for i in range(L)])
        self.d = torch.nn.Dropout(0.5)

    def forward(self, x, ei):
        for i, c in enumerate(self.cv[:-1]):
            x = F.elu(self.bn[i](c(self.d(x), ei)))
        return F.elu(self.bn[-1](self.cv[-1](self.d(x), ei)))


class BD(torch.nn.Module):
    def __init__(self, ic, h, oc, L):
        super().__init__()
        self.m = torch.nn.ModuleList([torch.nn.Linear(ic if i == 0 else h, oc if i == L - 1 else h) for i in range(L)])
        self.d = torch.nn.Dropout(0.5)

    def forward(self, z, e):
        x = z[e[0]] * z[e[1]]
        for m in self.m[:-1]:
            x = F.elu(m(self.d(x)))
        return self.m[-1](x)


def run_seed(seed, epochs=1000, p=0.3):
    pairs = common.load_ncdr()[0]
    msg, tpos = common.build_split(pairs, seed)
    m = np.asarray(msg); tp = np.asarray(tpos)
    P = common.ncdr_pos_set(pairs)
    tneg = common.sample_neg(P, tp.shape[1], seed + 7)
    torch.manual_seed(seed)
    e = BE(64, 128, 256, 2); d = BD(256, 64, 1, 2)
    opt = torch.optim.Adam(list(e.parameters()) + list(d.parameters()), lr=1e-3, weight_decay=5e-4)
    msgt = torch.tensor(m)
    nm = int(m.shape[1] * p)
    adj = torch.LongTensor([[a, b + common.NM] for a, b in pairs]).T
    full, _ = add_self_loops(adj)
    for _ in range(epochs):
        e.train(); d.train(); opt.zero_grad()
        perm = torch.randperm(m.shape[1])
        masked = msgt[:, perm[:nm]]
        z = e(feat, torch.cat([msgt, msgt.flip(0)], 1))
        neg = negative_sampling(full, num_nodes=common.NM + common.ND, num_neg_samples=nm)
        po = d(z, masked).view(-1); no = d(z, neg).view(-1)
        (F.binary_cross_entropy(po.sigmoid(), torch.ones_like(po))
         + F.binary_cross_entropy(no.sigmoid(), torch.zeros_like(no))).backward()
        torch.nn.utils.clip_grad_norm_(list(e.parameters()) + list(d.parameters()), 1.0)
        opt.step()
    e.eval(); d.eval()
    with torch.no_grad():
        z = e(feat, torch.cat([msgt, msgt.flip(0)], 1))
        pp = d(z, torch.tensor(tp)).view(-1).numpy()
        pn = d(z, torch.tensor(tneg)).view(-1).numpy()
    return common.full_metrics(pp, pn)


def main():
    ms = []
    for s in common.SEEDS:
        m = run_seed(s)
        ms.append(m)
        print(f"seed {s}: AUC {m['AUC']:.4f} AUPR {m['AUPR']:.4f} ACC {m['ACC']:.4f} "
              f"F1 {m['F1']:.4f} MCC {m['MCC']:.4f}")
    print("\nGAM-MDR (ncDR same-protocol) means:")
    for k in ("AUC", "AUPR", "ACC", "Precision", "Recall", "Specificity", "F1", "MCC"):
        v = np.array([m[k] for m in ms])
        print(f"  {k:12s} {v.mean():.4f} +- {v.std():.4f}")


if __name__ == "__main__":
    main()
