# -*- coding: utf-8 -*-
"""DLST-MDA baseline on the PLMF-MDA MDR/MDS benchmark (paper 5-fold protocol).

Run:
  python run_dlstmda_plmf.py --dataset MDR --split 5cv --seed 42
  python run_dlstmda_plmf.py --dataset MDS --split single --seed 0

Uses the official `GCNNetmuti` network (models_extra.py) unchanged; only the
featurisation is pointed at PLMF-MDA pairs (drug SMILES graph + SMILES char seq +
miRNA sequence), via the same helpers as the existing PLMF baseline suite.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGLoader

import common_extra as C
from models_extra import GCNNetmuti


def build_dataset(pairs, labels):
    data_list = []
    for (smiles, seq), y in zip(pairs, labels):
        _, features, edge_index = C.smile_to_graph(smiles)
        d = Data(x=torch.tensor(np.asarray(features), dtype=torch.float),
                 edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                 y=torch.tensor([float(y)], dtype=torch.float))
        d.target = torch.tensor(C.seq2int(seq), dtype=torch.long).unsqueeze(0)
        d.seqdrug = torch.tensor(C.smiles2int(smiles), dtype=torch.long).unsqueeze(0)
        data_list.append(d)
    return data_list


def run_fold(model, tr_loader, te_loader, device, epochs, lr, log, tag):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCELoss()
    best, last = {'AUC': -1}, None
    y_te = np.concatenate([d.y.view(-1).numpy() for d in te_loader.dataset])
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for data in tr_loader:
            data = data.to(device)
            opt.zero_grad()
            out = model(data)
            loss = lossf(out, data.y.view(-1, 1).float())
            loss.backward()
            opt.step()
            tot += loss.item() * data.num_graphs
        model.eval()
        ps = []
        with torch.no_grad():
            for data in te_loader:
                data = data.to(device)
                ps.append(model(data).view(-1).cpu().numpy())
        m = C.compute_metrics(y_te, np.concatenate(ps))
        m['epoch'] = ep
        last = m
        if m['AUC'] > best['AUC']:
            best = m
        if log and (ep % 10 == 0 or ep == 1):
            msg = (f'  {tag} ep{ep:03d} loss{tot/len(tr_loader.dataset):.4f} '
                   f'AUC{m["AUC"]:.4f} AUPR{m["AUPR"]:.4f} best{best["AUC"]:.4f}')
            print(msg, flush=True)
            log.write(msg + '\n')
            log.flush()
    return best, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['MDR', 'MDS'])
    ap.add_argument('--split', default='5cv', choices=['5cv', 'single'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--device', default=C.DEV)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    os.makedirs(C.RESULTS, exist_ok=True)
    os.makedirs(C.LOGS, exist_ok=True)
    folds = [int(x) for x in args.folds.split(',')]
    out_csv = args.out or os.path.join(
        C.RESULTS, f'dlstmda_{args.dataset}_{args.split}_seed{args.seed}.csv')
    log_path = os.path.join(C.LOGS, f'dlstmda_{args.dataset}_{args.split}_seed{args.seed}.log')

    rows = []
    with open(log_path, 'w') as log:
        log.write(f'=== DLST-MDA (official GCNNetmuti) {args.dataset} {args.split} '
                  f'seed={args.seed} epochs={args.epochs} lr={args.lr} batch={args.batch} ===\n')
        for fold in folds:
            t0 = time.time()
            tr_pairs, tr_y, te_pairs, te_y = C.get_fold(args.dataset, args.split, fold, args.seed)
            tr = build_dataset(tr_pairs, tr_y)
            te = build_dataset(te_pairs, te_y)
            tr_loader = PyGLoader(tr, batch_size=args.batch, shuffle=True)
            te_loader = PyGLoader(te, batch_size=args.batch, shuffle=False)
            C.seed_it(args.seed)
            model = GCNNetmuti().to(args.device)
            best, last = run_fold(model, tr_loader, te_loader, args.device, args.epochs,
                                  args.lr, log, f'dlstmda {args.dataset} {args.split} f{fold} s{args.seed}')
            rows.append(C.row_from_metrics(args.dataset, args.split, 'DLST-MDA', fold,
                                           args.seed, best, last))
            msg = (f'== DLST-MDA {args.dataset} {args.split} fold={fold} best_ep={best["epoch"]} '
                   f'best AUC={best["AUC"]:.4f} AUPR={best["AUPR"]:.4f} ACC={best["ACC"]:.4f} '
                   f'| last AUC={last["AUC"]:.4f} ({time.time()-t0:.0f}s)')
            print(msg, flush=True)
            log.write(msg + '\n')
            log.flush()

    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=C.csv_fields())
        w.writeheader()
        w.writerows(rows)
    C.summarize_rows(rows, f'DLST-MDA {args.dataset} {args.split}')
    print('saved', out_csv)


if __name__ == '__main__':
    main()
