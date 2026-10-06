"""Reproduce NDSGCL (local reimpl) on the ncDR split."""
import os
import random
import sys

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

DEV = "cuda:0"
EMB, N_LAYER, LR, EPOCH, BATCH = 64, 2, 0.01, 20, 2048
SSL_REG, PROTO_REG, TAU, K_CLUST, ALPHA, HYPER_LAYERS, REG = 1e-7, 8e-8, 0.1, 5, 1, 1, 1e-4


def seed_it(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class LGCN_Encoder(nn.Module):
    def __init__(self, nc, drug, emb, n_layers):
        super().__init__()
        self.lnc_emb = nn.Parameter(nn.init.xavier_uniform_(torch.empty(nc, emb)))
        self.drug_emb = nn.Parameter(nn.init.xavier_uniform_(torch.empty(drug, emb)))
        self.nc = nc; self.n_layers = n_layers

    def forward(self, adj):
        ego = torch.cat([self.lnc_emb, self.drug_emb], 0)
        embs = [ego]
        for _ in range(self.n_layers):
            ego = torch.sparse.mm(adj, ego); embs.append(ego)
        out = torch.stack(embs, dim=1).mean(dim=1)
        return out[:self.nc], out[self.nc:], embs


def bpr_loss(u, pos, neg):
    return -torch.log(1e-8 + torch.sigmoid((u * pos).sum(-1) - (u * neg).sum(-1))).mean()


def info_nce(v1, v2, tau):
    v1 = F.normalize(v1, dim=1); v2 = F.normalize(v2, dim=1)
    pos = torch.exp((v1 * v2).sum(-1) / tau)
    ttl = torch.exp(torch.matmul(v1, v2.t()) / tau).sum(-1)
    return -(torch.log(pos / ttl + 1e-8)).mean()


def ssl_layer_loss(ctx, init, u, i, nc):
    return SSL_REG * (info_nce(ctx[:nc][u], init[:nc][u], TAU)
                      + ALPHA * info_nce(ctx[nc:][i], init[nc:][i], TAU))


def kmeans_emb(emb, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(emb.cpu().numpy())
    return (torch.tensor(km.cluster_centers_, dtype=torch.float32, device=DEV),
            torch.tensor(km.labels_, dtype=torch.long, device=DEV))


def build_ncdr(seed):
    pairs = common.load_ncdr()[0]
    msg, tpos = common.build_split(pairs, seed)
    P = common.ncdr_pos_set(pairs)
    tneg = common.sample_neg(P, tpos.shape[1], seed + 7)
    nc, drug = common.NM, common.ND
    R = sp.coo_matrix((np.ones(msg.shape[1]), (msg[0], msg[1])), shape=(nc, drug)).tocsr()
    N = nc + drug
    A = sp.vstack([sp.hstack([sp.csr_matrix((nc, nc)), R]),
                   sp.hstack([R.T, sp.csr_matrix((drug, drug))])])
    A = (A != 0).astype(float) + sp.eye(N)
    deg = np.array(A.sum(1)).flatten(); dinv = np.power(deg, -0.5); dinv[np.isinf(dinv)] = 0
    nA = (sp.diags(dinv) @ A @ sp.diags(dinv)).tocoo()
    adjT = torch.sparse_coo_tensor(
        torch.tensor(np.vstack([nA.row, nA.col]), dtype=torch.long),
        torch.tensor(nA.data, dtype=torch.float32), (N, N)).coalesce().to(DEV)
    return nc, drug, adjT, list(zip(msg[0].tolist(), msg[1].tolist())), tpos, tneg


def run_seed(seed):
    seed_it(seed)
    nc, drug, adjT, pairs, tpos, tneg = build_ncdr(seed)
    model = LGCN_Encoder(nc, drug, EMB, N_LAYER).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=REG)
    n = len(pairs)
    for ep in range(EPOCH):
        with torch.no_grad():
            le, de, _ = model(adjT)
        lc, lk = kmeans_emb(le, K_CLUST); dc, dk = kmeans_emb(de, K_CLUST)
        random.shuffle(pairs)
        for b in range(0, n, BATCH):
            batch = pairs[b:b + BATCH]
            u = torch.tensor([p[0] for p in batch], device=DEV)
            i = torch.tensor([p[1] for p in batch], device=DEV)
            ineg = torch.randint(0, drug, (len(batch),), device=DEV)
            model.train(); opt.zero_grad()
            le2, de2, embs = model(adjT)
            rec = bpr_loss(le2[u], de2[i], de2[ineg])
            ssl = ssl_layer_loss(embs[HYPER_LAYERS * 2], embs[0], u, i, nc)
            proto = PROTO_REG * (info_nce(le2[u], lc[lk[u]], TAU) + info_nce(de2[i], dc[dk[i]], TAU))
            (rec + ssl + proto).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        le, de, _ = model(adjT)
        pp = (le[torch.tensor(tpos[0], device=DEV)] * de[torch.tensor(tpos[1], device=DEV)]).sum(-1).cpu().numpy()
        pn = (le[torch.tensor(tneg[0], device=DEV)] * de[torch.tensor(tneg[1], device=DEV)]).sum(-1).cpu().numpy()
    return common.full_metrics(pp, pn)


def main():
    ms = []
    for s in common.SEEDS:
        m = run_seed(s); ms.append(m)
        print(f"seed {s}: AUC {m['AUC']:.4f} AUPR {m['AUPR']:.4f} ACC {m['ACC']:.4f} F1 {m['F1']:.4f} MCC {m['MCC']:.4f}")
    print("\nNDSGCL (ncDR same-protocol) means:")
    for k in ("AUC", "AUPR", "ACC", "Precision", "Recall", "Specificity", "F1", "MCC"):
        v = np.array([m[k] for m in ms]); print(f"  {k:12s} {v.mean():.4f} +- {v.std():.4f}")


if __name__ == "__main__":
    main()
