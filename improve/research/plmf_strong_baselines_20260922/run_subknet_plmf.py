# -*- coding: utf-8 -*-
"""SubKNet (IEEE JBHI 2024) baseline on the PLMF-MDA MDR/MDS benchmark.

Run:
  python run_subknet_plmf.py --dataset MDR --split 5cv --seed 42

Uses SubKNet's official code path (`subgraph_extraction.links2subgraphs`,
`model.Model`, `prediction.preds`) unchanged -- only the data is the PLMF-MDA
benchmark, formatted as <dataset>/{sims_m.txt, sims_d.txt, inters.txt}:

  sims_m : cosine similarity of 2/3-mer miRNA frequency vectors
  sims_d : cosine similarity of Morgan (r=2, 2048-bit) drug fingerprints
  inters : binary association matrix built from the *training-fold* positives

Per fold the observed graph masks the held-out test positives, and we keep the
test-best epoch (identical protocol to the other PLMF baselines).
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import common_extra as C  # noqa: E402

CODE = os.path.join(HERE, 'repos', 'SubKNet', 'code')
SUBDATA = os.path.join(HERE, 'repos', 'SubKNet', 'data')

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=['MDR', 'MDS'])
ap.add_argument('--split', default='5cv', choices=['5cv', 'single'])
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--folds', default='0,1,2,3,4')
ap.add_argument('--epochs', type=int, default=60)
ap.add_argument('--lr', type=float, default=1e-2)
ap.add_argument('--batch', type=int, default=128)
ap.add_argument('--max_step', type=int, default=2)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--out', default='')
a = ap.parse_args()

TAG = f'PLMF_{a.dataset}'
os.environ['CUDA_VISIBLE_DEVICES'] = a.device.split(':')[-1]
os.chdir(CODE)
sys.path.insert(0, CODE)
# SubKNet modules parse sys.argv (prediction.py at import time); fix it up first.
sys.argv = ['subknet', '--dataset', TAG, '--types', 'hete', '--epochs', str(a.epochs),
            '--lr', str(a.lr), '--batch_size', str(a.batch), '--fixs', '0',
            '--max_step', str(a.max_step), '--gcn_layer', '1', '--dropout_rate', '0.2',
            '--lambda1', '0.5', '--lambda2', '0.5', '--lambda3', '0.5']

import torch  # noqa: E402
from parameters import args_parser  # noqa: E402
from utils import comp_feat  # noqa: E402
from subgraph_extraction import links2subgraphs  # noqa: E402
from model import Model  # noqa: E402
import prediction as _prediction  # noqa: E402
_prediction.ProgressBar.update = lambda self, *x, **k: None
from prediction import preds  # noqa: E402

args = args_parser()
DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def cos_sim(x):
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    return x @ x.T


def write_sims(dataset):
    xr, xd, mirnas, drugs = C.graph_cache(dataset)
    d = os.path.join(SUBDATA, TAG)
    os.makedirs(d, exist_ok=True)
    np.savetxt(os.path.join(d, 'sims_m.txt'), cos_sim(xr.astype(np.float64)), delimiter='\t')
    np.savetxt(os.path.join(d, 'sims_d.txt'), cos_sim(xd.astype(np.float64)), delimiter='\t')
    return mirnas, drugs


def full_metrics(out2, yy):
    """out2: (N,2) logits -> class-1 prob; reuse the suite's metric definition."""
    sc = np.asarray(out2, dtype=float)
    sc = sc[:, 1] / np.clip(sc.sum(1), 1e-12, None)
    m = C.compute_metrics(np.asarray(yy), sc)
    return m


def main():
    os.makedirs(C.RESULTS, exist_ok=True)
    os.makedirs(C.LOGS, exist_ok=True)
    folds = [int(x) for x in a.folds.split(',')]
    out_csv = a.out or os.path.join(
        C.RESULTS, f'subknet_{a.dataset}_{a.split}_seed{a.seed}.csv')
    log_path = os.path.join(C.LOGS, f'subknet_{a.dataset}_{a.split}_seed{a.seed}.log')

    mirnas, drugs = write_sims(a.dataset)
    midx = {s: i for i, s in enumerate(mirnas)}
    didx = {s: j for j, s in enumerate(drugs)}
    n_m, n_d = len(mirnas), len(drugs)

    rows = []
    with open(log_path, 'w') as log:
        log.write(f'=== SubKNet (official) {a.dataset} {a.split} seed={a.seed} '
                  f'epochs={a.epochs} lr={a.lr} batch={a.batch} max_step={a.max_step} '
                  f'n_mirna={n_m} n_drug={n_d} ===\n')

        def w(msg):
            print(msg, flush=True)
            log.write(msg + '\n')
            log.flush()

        for fold in folds:
            t0 = time.time()
            tr_pairs, tr_y, te_pairs, te_y = C.get_fold(a.dataset, a.split, fold, a.seed)

            def idx_of(pairs):
                return np.array([midx[s] for _, s in pairs]), np.array([didx[sm] for sm, _ in pairs])

            tr_m, tr_d = idx_of(tr_pairs)
            te_m, te_d = idx_of(te_pairs)
            tr_y = np.asarray(tr_y).astype(int)
            te_y = np.asarray(te_y).astype(int)

            MD = np.zeros((n_m, n_d), dtype=np.float64)
            pos = tr_y == 1
            MD[tr_m[pos], tr_d[pos]] = 1.0
            train_pos = np.where(MD == 1)
            train_zero = (tr_m[~pos], tr_d[~pos])
            test_pos = (te_m[te_y == 1], te_d[te_y == 1])
            test_neg = (te_m[te_y == 0], te_d[te_y == 0])

            featM, featD, featM_sim, featD_sim = comp_feat(MD, TAG)
            tr, va, te, max_n = links2subgraphs(MD, train_pos, train_zero, None, None,
                                                test_pos, test_neg, h=1,
                                                featM=featM, featD=featD)
            w(f'  fold{fold}: train={len(tr)} test={len(te)} subgraph_max_n={max_n}')
            size_sub = int(np.mean([x[0].shape[0] for x in tr]))

            C.seed_it(a.seed)
            model = Model(featM_sim.shape[1], featD_sim.shape[1], tr[0][1].shape[1], 2, args,
                          max_step=args.max_step, dropout_rate=args.dropout_rate,
                          size_graph_filter=[size_sub]).to(DEV)
            opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=5e-4)
            crit = torch.nn.CrossEntropyLoss()

            yy_tr = [x[3] for x in tr]
            adj_tr = [x[0] for x in tr]
            f_tr = [x[1] for x in tr]
            n_tr = [x[2] for x in tr]
            yy_te = [x[3] for x in te]
            adj_te = [x[0] for x in te]
            f_te = [x[1] for x in te]
            n_te = [x[2] for x in te]

            best = (-1, None, None, None)
            last = None
            for ep in range(a.epochs):
                model.train()
                _, _, _, _ = preds(MD, adj_tr, f_tr, n_tr, yy_tr, featM_sim, featD_sim,
                                   a.batch, ep, model, optimizer=opt, criterion=crit)
                model.eval()
                _, out_te, tg_te, cv = preds(MD, adj_te, f_te, n_te, yy_te, featM_sim, featD_sim,
                                             a.batch, ep, model, optimizer=None, criterion=crit)
                last = (cv, out_te, tg_te)
                if cv[1] > best[0]:
                    best = (cv[1], out_te, tg_te, cv)
                if (ep + 1) % 10 == 0 or ep == 0:
                    w(f'  subknet {a.dataset} {a.split} f{fold} ep{ep+1:03d} '
                      f'AUC{cv[1]:.4f} AUPR{cv[0]:.4f} best{best[0]:.4f}')

            best_m = full_metrics(best[1], best[2])
            best_m['epoch'] = 0  # epoch index not tracked for SubKNet (early-best)
            last_m = full_metrics(last[1], last[2])
            last_m['epoch'] = a.epochs
            rows.append(C.row_from_metrics(a.dataset, a.split, 'SubKNet', fold,
                                           a.seed, best_m, last_m))
            w(f'== SubKNet {a.dataset} {a.split} fold={fold} best AUC={best_m["AUC"]:.4f} '
              f'AUPR={best_m["AUPR"]:.4f} ACC={best_m["ACC"]:.4f} '
              f'| last AUC={last_m["AUC"]:.4f} ({time.time()-t0:.0f}s)')

    with open(out_csv, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=C.csv_fields())
        wr.writeheader()
        wr.writerows(rows)
    C.summarize_rows(rows, f'SubKNet {a.dataset} {a.split}')
    print('saved', out_csv)


if __name__ == '__main__':
    main()
