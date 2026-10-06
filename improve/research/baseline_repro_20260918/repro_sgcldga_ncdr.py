"""Reproduce SGCLDGA (BIB 2024) on the ncDR same-protocol benchmark."""
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch
import torch.utils.data as data

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "work", "sgcldga")
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import common  # noqa: E402
from model import SGCLDGA  # noqa: E402
from utils import TrnData, scipy_sparse_mat_to_torch_sparse_tensor  # noqa: E402

DEVICE = "cuda:0"
D = 1024
L = 4
TEMP = 0.8
LAM1 = 1e-3
LAM2 = 1e-5
DROPOUT = 0.1
EPOCHS = int(os.environ.get("EPOCHS", "50"))
LR = 1e-3
Q = 5
INTER_BATCH = 4096
BATCH = 256
METRICS = []


def run_seed(pairs, seed):
    msg, tpos = common.build_split(pairs, seed)
    P = common.ncdr_pos_set(pairs)
    train = sp.coo_matrix(
        (np.ones(msg.shape[1]), (msg[0], msg[1])), shape=(common.NM, common.ND)
    ).astype(np.float64)
    train_csr = (train != 0).astype(np.float32)
    rowD = np.asarray(train.sum(1)).squeeze()
    colD = np.asarray(train.sum(0)).squeeze()
    rowD[rowD == 0] = 1.0
    colD[colD == 0] = 1.0
    for i in range(len(train.data)):
        train.data[i] = train.data[i] / pow(rowD[train.row[i]] * colD[train.col[i]], 0.5)
    train_coo = train.tocoo()
    tr_data = TrnData(train_coo)
    loader = data.DataLoader(tr_data, batch_size=INTER_BATCH, shuffle=True, num_workers=0)

    adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().to(DEVICE)
    svd_u, s, svd_v = torch.svd_lowrank(adj_norm, q=Q)
    u_mul_s = svd_u @ torch.diag(s)
    v_mul_s = svd_v @ torch.diag(s)

    torch.manual_seed(seed)
    model = SGCLDGA(adj_norm.shape[0], adj_norm.shape[1], D, u_mul_s, v_mul_s,
                    svd_u.T, svd_v.T, train_csr, adj_norm, L, TEMP, LAM1, LAM2,
                    DROPOUT, BATCH, DEVICE)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), weight_decay=0, lr=LR)
    model.train()
    for ep in range(EPOCHS):
        tr_data.neg_sampling()
        tot = 0.0
        for uids, pos, neg in loader:
            uids = uids.long().to(DEVICE)
            pos = pos.long().to(DEVICE)
            neg = neg.long().to(DEVICE)
            iids = torch.cat([pos, neg], dim=0)
            opt.zero_grad()
            loss, _, _ = model(uids, iids, pos, neg)
            loss.backward()
            opt.step()
            tot += loss.item()
        if (ep + 1) % 10 == 0:
            print(f"  seed {seed} epoch {ep+1} loss {tot/len(loader):.4f}")

    model.dropout = 0.0
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, dtype=torch.long, device=DEVICE)
        model(torch.arange(common.NM, device=DEVICE), dummy, dummy, dummy)
        Eg, Ed = model.E_g, model.E_d
        p0 = torch.tensor(tpos[0], dtype=torch.long, device=DEVICE)
        p1 = torch.tensor(tpos[1], dtype=torch.long, device=DEVICE)
        pos_scores = (Eg[p0] * Ed[p1]).sum(1).cpu().numpy()
    tneg = common.sample_neg(P, tpos.shape[1], seed + 7)
    with torch.no_grad():
        n0 = torch.tensor(tneg[0], dtype=torch.long, device=DEVICE)
        n1 = torch.tensor(tneg[1], dtype=torch.long, device=DEVICE)
        neg_scores = (Eg[n0] * Ed[n1]).sum(1).cpu().numpy()
    m = common.full_metrics(pos_scores, neg_scores)
    METRICS.append(m)
    return m["AUC"], m["AUPR"]


def main():
    pairs, _, _ = common.load_ncdr()
    seeds = common.SEEDS
    if os.environ.get("QUICK"):
        seeds = [42]
    aucs, auprs = [], []
    for s in seeds:
        a, p = run_seed(pairs, s)
        print(f"seed {s}: AUC {a:.4f} AUPR {p:.4f}")
        aucs.append(a)
        auprs.append(p)
    common.summarize("SGCLDGA (ncDR same-protocol)", aucs, auprs,
                     os.path.join(HERE, "results_sgcldga_ncdr.csv"))
    print("full metrics @bestF1:",
          {k: round(float(np.mean([mm[k] for mm in METRICS])), 4)
           for k in ("ACC", "Precision", "Recall", "Specificity", "F1", "MCC")})


if __name__ == "__main__":
    main()
