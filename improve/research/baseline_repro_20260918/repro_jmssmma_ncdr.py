"""Reproduce JMSSMMA (Briefings in Bioinformatics 2022) on ncDR same-protocol.

JMSSMMA is a graph masked autoencoder for small-molecule/miRNA associations.
We feed it the ncDR bipartite matrix plus GIP self-similarity node features
computed from the TRAINING adjacency only (no test leakage), then evaluate on
the shared 80/20 split.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repos", "JMSSMMA")
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import common  # noqa: E402
from mask import Mask  # noqa: E402
from model import DegreeDecoder, EdgeDecoder, GMAE, GNNEncoder  # noqa: E402
from utils import set_seed  # noqa: E402

DIM = 1024
EPOCHS = int(os.environ.get("EPOCHS", "1000"))
METRICS = []


def gip(mat):
    """Gaussian interaction profile similarity (rows)."""
    r = (mat ** 2).sum() / mat.shape[0]
    if r == 0:
        r = 1.0
    m = mat @ mat.T
    d = (mat ** 2).sum(1)
    dist = d[:, None] + d[None, :] - 2 * m
    S = np.exp(-dist / r)
    np.fill_diagonal(S, 1.0)
    return S


def pad(feat, dim):
    out = np.zeros((feat.shape[0], dim), dtype=np.float32)
    k = min(dim, feat.shape[1])
    out[:, :k] = feat[:, :k]
    return out


def run_seed(pairs, seed):
    msg, tpos = common.build_split(pairs, seed)
    P = common.ncdr_pos_set(pairs)
    train = np.zeros((common.NM, common.ND), dtype=np.float64)
    train[msg[0], msg[1]] = 1.0
    m_sim = gip(train)            # miRNA x miRNA
    d_sim = gip(train.T)          # drug x drug
    x = torch.tensor(np.vstack([pad(m_sim, DIM), pad(d_sim, DIM)]), dtype=torch.float32)
    edge = torch.tensor(np.stack([np.r_[msg[0], tpos[0]],
                                  np.r_[msg[1], tpos[1]] + common.NM]), dtype=torch.long)
    train_edge = edge[:, : msg.shape[1]]

    set_seed(seed)
    dev = "cuda:0"
    encoder = GNNEncoder(in_channels=DIM, hidden_channels=64, out_channels=128)
    model = GMAE(encoder, EdgeDecoder(128, 64, 1), DegreeDecoder(128, 64, 1), Mask(p=0.4)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
    x = x.to(dev)
    train_edge = train_edge.to(dev)

    class D:
        pass

    data = D()
    data.x = x
    data.edge_index = train_edge
    data.num_nodes = x.size(0)

    for ep in range(EPOCHS):
        model.train()
        model.train_epoch(data, opt, alpha=0.007)
        if (ep + 1) % 200 == 0:
            print(f"  seed {seed} epoch {ep+1}")

    model.eval()
    with torch.no_grad():
        z = model.encoder(x, train_edge)
        p0 = torch.tensor(tpos[0], dtype=torch.long, device=dev)
        p1 = torch.tensor(tpos[1] + common.NM, dtype=torch.long, device=dev)
        pos_pred = model.batch_predict(z, torch.stack([p0, p1])).cpu().numpy()
        tneg = common.sample_neg(P, tpos.shape[1], seed + 7)
        n0 = torch.tensor(tneg[0], dtype=torch.long, device=dev)
        n1 = torch.tensor(tneg[1] + common.NM, dtype=torch.long, device=dev)
        neg_pred = model.batch_predict(z, torch.stack([n0, n1])).cpu().numpy()
    m = common.full_metrics(pos_pred, neg_pred)
    METRICS.append(m)
    return m["AUC"], m["AUPR"]

def main():
    pairs, _, _ = common.load_ncdr()
    seeds = [42] if os.environ.get("QUICK") else common.SEEDS
    aucs, auprs = [], []
    for s in seeds:
        a, p = run_seed(pairs, s)
        print(f"seed {s}: AUC {a:.4f} AUPR {p:.4f}")
        aucs.append(a)
        auprs.append(p)
    common.summarize("JMSSMMA (ncDR same-protocol)", aucs, auprs,
                     os.path.join(HERE, "results_jmssmma_ncdr.csv"))
    print("full metrics @bestF1:",
          {k: round(float(np.mean([mm[k] for mm in METRICS])), 4)
           for k in ("ACC", "Precision", "Recall", "Specificity", "F1", "MCC")})


if __name__ == "__main__":
    main()
