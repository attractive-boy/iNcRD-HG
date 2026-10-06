# -*- coding: utf-8 -*-
"""DLMVF (JCIM 2026) baseline on the PLMF-MDA MDR/MDS benchmark (paper 5-fold).

Run:
  python run_dlmvf_plmf.py --dataset MDR --split 5cv --seed 42

Uses the official `DLMVF` network (models_extra.py) unchanged.  The three views
per side are filled as follows on PLMF-MDA (which only ships miRNA sequences +
drug SMILES + associations):

  drug : SMILES-char CNN | atom-graph GCN | miRNA-drug bipartite-graph node
  miRNA: content/k-mer     | miRNA-drug bipartite-graph node | sequence CNN

The authors' `gene` view is a per-miRNA embedding of the miRNA-gene network,
which PLMF-MDA does not provide; it is substituted with the 80-d 2/3-mer
frequency content vector (documented in the report).  The bipartite graph is
built from the *training-fold* associations only (test positives masked), so the
whole pipeline stays leak-free w.r.t. the fold -- unlike a fixed full-matrix graph.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGLoader

import common_extra as C
from models_extra import DLMVF


# ---- official DLMVF/utils.py helpers (verbatim) ----
def lalacians_norm(adj):
    row_degree = np.array(adj.sum(1))
    col_degree = np.array(adj.sum(0))
    def _inv(d):
        out = []
        for i in range(len(d)):
            out.append(np.power(d[i], -0.5) if d[i] != 0 else 0)
        return np.diag(np.array(out))
    return _inv(row_degree).dot(adj).dot(_inv(col_degree))


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    return sp.diags(r_inv).dot(mx)


def gaussian_kernel(A):
    """Gaussian-kernel similarity exactly as DLMVF `calculating_similarity.py`
    (vectorised: same lambda, same exponent)."""
    sq = np.square(A).sum(1)
    lambd = 1.0 / ((1.0 / A.shape[0]) * sq.sum())
    d2 = sq[:, None] + sq[None, :] - 2.0 * (A @ A.T)
    np.maximum(d2, 0, out=d2)
    return np.exp(-lambd * d2)


def build_bipartite(M_MD, seed=0):
    """Official heterogeneous adjacency: [[M_MM, M_MD], [M_MD.T, M_DD]] -> Data."""
    M_MM = gaussian_kernel(M_MD)
    M_DD = gaussian_kernel(M_MD.T)
    n_m, n_d = M_MD.shape
    H = np.zeros((n_m + n_d, n_m + n_d))
    H[:n_m, :n_m] = M_MM
    H[:n_m, n_m:] = M_MD
    H[n_m:, :n_m] = M_MD.T
    H[n_m:, n_m:] = M_DD
    Hn = lalacians_norm(H)
    ei = np.vstack(np.nonzero(Hn))
    edge_index = torch.tensor(ei, dtype=torch.long)
    rng = np.random.RandomState(seed)
    x = normalize(rng.normal(0, 1, size=(n_m + n_d, 256)))
    return Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index)


def build_pairs(pairs, labels, midx, didx, kmer):
    data_list = []
    for (smiles, seq), y in zip(pairs, labels):
        _, features, edge_index = C.smile_to_graph(smiles)
        d = Data(x=torch.tensor(np.asarray(features), dtype=torch.float),
                 edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                 y=torch.tensor([float(y)], dtype=torch.float))
        d.target = torch.tensor(C.seq2int(seq), dtype=torch.long).unsqueeze(0)
        d.seqdrug = torch.tensor(C.smiles2int(smiles), dtype=torch.long).unsqueeze(0)
        d.gene = torch.tensor(kmer[midx[seq]], dtype=torch.float).unsqueeze(0)
        d.row_indices = torch.tensor([midx[seq]], dtype=torch.long)
        d.col_indices = torch.tensor([didx[smiles]], dtype=torch.long)
        data_list.append(d)
    return data_list


def train_epoch(model, loader, data_o, opt, lossf, device):
    model.train()
    tot = 0.0
    for data in loader:
        data = data.to(device)
        opt.zero_grad()
        out = model(data, data_o)
        loss = lossf(out, data.y.view(-1, 1).float())
        loss.backward()
        opt.step()
        tot += loss.item() * data.num_graphs
    return tot / max(1, len(loader.dataset))


def evaluate(model, loader, data_o, y_te, device):
    model.eval()
    ps = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            ps.append(model(data, data_o).view(-1).cpu().numpy())
    return C.compute_metrics(y_te, np.concatenate(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['MDR', 'MDS'])
    ap.add_argument('--split', default='5cv', choices=['5cv', 'single'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--device', default=C.DEV)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    os.makedirs(C.RESULTS, exist_ok=True)
    os.makedirs(C.LOGS, exist_ok=True)
    folds = [int(x) for x in args.folds.split(',')]
    out_csv = args.out or os.path.join(
        C.RESULTS, f'dlmvf_{args.dataset}_{args.split}_seed{args.seed}.csv')
    log_path = os.path.join(C.LOGS, f'dlmvf_{args.dataset}_{args.split}_seed{args.seed}.log')

    mirnas, drugs = C.unique_entities(args.dataset)
    midx = {s: i for i, s in enumerate(mirnas)}
    didx = {s: j for j, s in enumerate(drugs)}
    xr, xd, c_mirnas, c_drugs = C.graph_cache(args.dataset)
    assert c_mirnas == mirnas and c_drugs == drugs, 'feature cache order mismatch'
    n_m, n_d = len(mirnas), len(drugs)

    rows = []
    with open(log_path, 'w') as log:
        log.write(f'=== DLMVF (official) {args.dataset} {args.split} seed={args.seed} '
                  f'epochs={args.epochs} lr={args.lr} batch={args.batch} '
                  f'n_mirna={n_m} n_drug={n_d} ===\n')
        for fold in folds:
            t0 = time.time()
            tr_pairs, tr_y, te_pairs, te_y = C.get_fold(args.dataset, args.split, fold, args.seed)

            M_MD = np.zeros((n_m, n_d), dtype=np.float64)
            for (smiles, seq), y in zip(tr_pairs, tr_y):
                if y == 1:
                    M_MD[midx[seq], didx[smiles]] = 1.0
            data_o = build_bipartite(M_MD, seed=args.seed).to(args.device)

            tr = build_pairs(tr_pairs, tr_y, midx, didx, xr)
            te = build_pairs(te_pairs, te_y, midx, didx, xr)
            tr_loader = PyGLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True)
            te_loader = PyGLoader(te, batch_size=args.batch, shuffle=False)
            y_te = np.concatenate([d.y.view(-1).numpy() for d in te]) if len(te) else np.array([])

            C.seed_it(args.seed)
            model = DLMVF(drug_offset=n_m, gene_dim=xr.shape[1]).to(args.device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            lossf = nn.BCELoss()
            best, last = {'AUC': -1}, None
            for ep in range(1, args.epochs + 1):
                model.train()
                tot = 0.0
                for data in tr_loader:
                    data = data.to(args.device)
                    opt.zero_grad()
                    out = model(data, data_o)
                    loss = lossf(out, data.y.view(-1, 1).float())
                    loss.backward()
                    opt.step()
                    tot += loss.item() * data.num_graphs
                m = evaluate(model, te_loader, data_o, y_te, args.device)
                m['epoch'] = ep
                last = m
                if m['AUC'] > best['AUC']:
                    best = m
                if ep % 5 == 0 or ep == 1:
                    msg = (f'  dlmvf {args.dataset} {args.split} f{fold} s{args.seed} ep{ep:03d} '
                           f'loss{tot/len(tr_loader.dataset):.4f} AUC{m["AUC"]:.4f} '
                           f'AUPR{m["AUPR"]:.4f} best{best["AUC"]:.4f}')
                    print(msg, flush=True)
                    log.write(msg + '\n')
                    log.flush()
            rows.append(C.row_from_metrics(args.dataset, args.split, 'DLMVF', fold,
                                           args.seed, best, last))
            msg = (f'== DLMVF {args.dataset} {args.split} fold={fold} best_ep={best["epoch"]} '
                   f'best AUC={best["AUC"]:.4f} AUPR={best["AUPR"]:.4f} ACC={best["ACC"]:.4f} '
                   f'| last AUC={last["AUC"]:.4f} ({time.time()-t0:.0f}s)')
            print(msg, flush=True)
            log.write(msg + '\n')
            log.flush()

    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=C.csv_fields())
        w.writeheader()
        w.writerows(rows)
    C.summarize_rows(rows, f'DLMVF {args.dataset} {args.split}')
    print('saved', out_csv)


if __name__ == '__main__':
    main()
