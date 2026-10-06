# -*- coding: utf-8 -*-
"""Reproduce PLMF-MDA's Fig 3 baselines (GraphDTA / SubMDTA / ML-DTI) on the
MDRdataset / MDSdataset, using the model definitions from the DLST-MDA
reproduction (`../dlstmda_baselines_20260919/models.py`).

Usage:
  python run_bench_plmf.py --model graphdta --dataset MDR --split 5cv --seed 42
  python run_bench_plmf.py --model mldti    --dataset MDS --split single --seed 0
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGLoader

HERE = os.path.dirname(os.path.abspath(__file__))
DLST = os.path.join(os.path.dirname(HERE), 'dlstmda_baselines_20260919')
sys.path.insert(0, DLST)
sys.path.insert(0, HERE)

from common import DEV, RESULTS, LOGS, seed_it, seq2int, smiles2int, compute_metrics  # noqa: E402
from models import GraphDTA_GCN, SubMDTA, MLDTI, build_graph_dataset  # noqa: E402
import plmf_data  # noqa: E402

RESULTS = os.path.join(HERE, 'results')
LOGS = os.path.join(HERE, 'logs')
METHOD = {'graphdta': 'GraphDTA', 'submdta': 'SubMDTA', 'mldti': 'ML-DTI'}
DEFAULTS = {'graphdta': (150, 5e-4, 128), 'submdta': (150, 5e-4, 128),
            'mldti': (150, 1e-3, 256)}


def get_fold(dataset, split, fold, seed):
    if split == '5cv':
        return plmf_data.load_fold_5cv(dataset, fold, seed)
    return plmf_data.load_single_split(dataset, seed)


def train_graph(model, tr_loader, te_loader, device, epochs, lr, log=None, tag=''):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    best, last = {'AUC': -1}, None
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for data in tr_loader:
            data = data.to(device)
            opt.zero_grad()
            out = model(data).view(-1)
            loss = lossf(out, data.y.view(-1))
            loss.backward()
            opt.step()
            tot += loss.item() * data.num_graphs
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for data in te_loader:
                data = data.to(device)
                out = torch.sigmoid(model(data).view(-1)).cpu().numpy()
                ys.append(data.y.view(-1).cpu().numpy())
                ps.append(out)
        m = compute_metrics(np.concatenate(ys), np.concatenate(ps))
        m['epoch'] = ep
        last = m
        if m['AUC'] > best['AUC']:
            best = m
        if log and (ep % 20 == 0 or ep == 1):
            msg = (f"  {tag} ep{ep:03d} loss{tot/len(tr_loader.dataset):.4f} "
                   f"AUC{m['AUC']:.4f} AUPR{m['AUPR']:.4f} best{best['AUC']:.4f}")
            print(msg, flush=True)
            log.write(msg + '\n')
            log.flush()
    return best, last


def train_tensor(model, tr_xy, te_xy, device, epochs, lr, batch, log=None, tag=''):
    prot_tr, drug_tr, y_tr = tr_xy
    prot_te, drug_te, y_te = te_xy
    ds = torch.utils.data.TensorDataset(prot_tr, drug_tr, y_tr)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    best, last = {'AUC': -1}, None
    y_np = y_te.view(-1).cpu().numpy()
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for pb, db, yb in loader:
            pb, db, yb = pb.to(device), db.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(pb, db).view(-1), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(prot_te.to(device), drug_te.to(device)).view(-1)).cpu().numpy()
        m = compute_metrics(y_np, p)
        m['epoch'] = ep
        last = m
        if m['AUC'] > best['AUC']:
            best = m
        if log and (ep % 20 == 0 or ep == 1):
            msg = (f"  {tag} ep{ep:03d} loss{tot/len(y_tr):.4f} "
                   f"AUC{m['AUC']:.4f} AUPR{m['AUPR']:.4f} best{best['AUC']:.4f}")
            print(msg, flush=True)
            log.write(msg + '\n')
            log.flush()
    return best, last


def run_fold_graph(model_name, dataset, split, fold, seed, device, epochs, lr, batch, log):
    tr_pairs, tr_y, te_pairs, te_y = get_fold(dataset, split, fold, seed)
    tr = build_graph_dataset(tr_pairs, tr_y)
    te = build_graph_dataset(te_pairs, te_y)
    tr_loader = PyGLoader(tr, batch_size=batch, shuffle=True)
    te_loader = PyGLoader(te, batch_size=batch, shuffle=False)
    seed_it(seed)
    model = GraphDTA_GCN() if model_name == 'graphdta' else SubMDTA()
    model = model.to(device)
    return train_graph(model, tr_loader, te_loader, device, epochs, lr, log,
                       tag=f'{model_name} {dataset} {split} f{fold} s{seed}')


def run_fold_mldti(dataset, split, fold, seed, device, epochs, lr, batch, log):
    tr_pairs, tr_y, te_pairs, te_y = get_fold(dataset, split, fold, seed)

    def prep(pairs, ys):
        prot = torch.tensor(np.stack([seq2int(s) for _, s in pairs]), dtype=torch.long)
        drug = torch.tensor(np.stack([smiles2int(a) for a, _ in pairs]), dtype=torch.long)
        return prot, drug, torch.tensor(ys, dtype=torch.float)

    seed_it(seed)
    model = MLDTI().to(device)
    return train_tensor(model, prep(tr_pairs, tr_y), prep(te_pairs, te_y),
                        device, epochs, lr, batch, log,
                        tag=f'mldti {dataset} {split} f{fold} s{seed}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(METHOD))
    ap.add_argument('--dataset', required=True, choices=['MDR', 'MDS'])
    ap.add_argument('--split', default='5cv', choices=['5cv', 'single'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--folds', type=str, default='0,1,2,3,4')
    ap.add_argument('--epochs', type=int, default=0)
    ap.add_argument('--lr', type=float, default=0.0)
    ap.add_argument('--batch', type=int, default=0)
    ap.add_argument('--device', default=DEV)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    epochs = args.epochs or DEFAULTS[args.model][0]
    lr = args.lr or DEFAULTS[args.model][1]
    batch = args.batch or DEFAULTS[args.model][2]
    folds = [int(x) for x in args.folds.split(',')]
    device = args.device

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    out_csv = args.out or os.path.join(RESULTS, f'{args.model}_{args.dataset}_{args.split}_seed{args.seed}.csv')
    log_path = os.path.join(LOGS, f'{args.model}_{args.dataset}_{args.split}_seed{args.seed}.log')
    fields = ['dataset', 'split', 'method', 'fold', 'seed', 'auc', 'aupr', 'acc', 'precision',
              'recall', 'specificity', 'f1', 'mcc', 'best_epoch', 'last_auc', 'last_aupr']
    rows = []
    with open(log_path, 'w') as log:
        log.write(f'=== {args.model} {args.dataset} {args.split} seed={args.seed} '
                  f'epochs={epochs} lr={lr} batch={batch} ===\n')
        for fold in folds:
            t0 = time.time()
            if args.model in ('graphdta', 'submdta'):
                best, last = run_fold_graph(args.model, args.dataset, args.split, fold, args.seed,
                                            device, epochs, lr, batch, log)
            else:
                best, last = run_fold_mldti(args.dataset, args.split, fold, args.seed,
                                            device, epochs, lr, batch, log)
            rows.append({'dataset': args.dataset, 'split': args.split, 'method': METHOD[args.model],
                         'fold': fold, 'seed': args.seed, 'auc': best['AUC'], 'aupr': best['AUPR'],
                         'acc': best['ACC'], 'precision': best['Precision'], 'recall': best['Recall'],
                         'specificity': best['Specificity'], 'f1': best['F1'], 'mcc': best['MCC'],
                         'best_epoch': best['epoch'], 'last_auc': last['AUC'], 'last_aupr': last['AUPR']})
            msg = (f'== {args.model} {args.dataset} {args.split} fold={fold} best_ep={best["epoch"]} '
                   f'best AUC={best["AUC"]:.4f} AUPR={best["AUPR"]:.4f} ACC={best["ACC"]:.4f} '
                   f'| last AUC={last["AUC"]:.4f} ({time.time()-t0:.0f}s)')
            print(msg, flush=True)
            log.write(msg + '\n')
            log.flush()
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    import statistics as st
    print(f'\n=== {METHOD[args.model]} {args.dataset} {args.split} ({len(rows)} folds) ===')
    for k in ['auc', 'aupr', 'acc', 'f1', 'mcc']:
        v = [r[k] for r in rows]
        sd = st.stdev(v) if len(v) > 1 else 0.0
        print(f'  {k}: {st.mean(v):.4f} +/- {sd:.4f}')
    print('saved', out_csv)


if __name__ == '__main__':
    main()
