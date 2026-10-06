# -*- coding: utf-8 -*-
"""Train + evaluate HGNNLDA (PyTorch port) on the RNAactDrug benchmark.

The official repo ships only the training graph (main.py has no evaluator).  We keep
the training recipe verbatim (HGNNLDA.conf) and add two standard evaluation protocols
used throughout this project:

  balanced : test positives vs. an equal number of random unknown pairs
  global   : test positives vs. all unknown pairs (ranked)

Run:
  /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python run_hgnnlda.py --fold 0 --seed 1
"""
import argparse
import csv
import os
import statistics as st
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

import hgnnlda_data as D
from hgnnlda_model import HGNNLDA, hypergraph_adjacency, bpr_loss, weight_reg

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')

# HGNNLDA.conf
EMB = 64
N_LAYER = 3
MAX_ITER = 100
BATCH = 2048
LR = 0.005
REG_U = 0.0001
DROPOUT = 0.1


def train_model(pos_tr, n_l, n_d, device, seed, propagate='e0'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    A = D.train_adjacency(pos_tr, n_l, n_d)
    A_u, A_i = hypergraph_adjacency(A)
    A_u, A_i = A_u.to(device), A_i.to(device)
    model = HGNNLDA(n_l, n_d, EMB, N_LAYER, DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    per_lncrna = {}
    for u, v in pos_tr:
        per_lncrna.setdefault(int(u), set()).add(int(v))
    drugs = np.arange(n_d)
    rng = np.random.default_rng(seed)
    n = len(pos_tr)

    model.train()
    for it in range(MAX_ITER):
        perm = rng.permutation(n)
        for b0 in range(0, n, BATCH):
            bidx = perm[b0:b0 + BATCH]
            u_idx = pos_tr[bidx, 0]
            i_idx = pos_tr[bidx, 1]
            j_idx = np.empty(len(bidx), dtype=np.int64)
            for k, u in enumerate(u_idx):
                s = per_lncrna.get(int(u), set())
                j = int(rng.integers(0, n_d))
                while j in s:
                    j = int(rng.integers(0, n_d))
                j_idx[k] = j
            ut = torch.tensor(u_idx, dtype=torch.long, device=device)
            it_ = torch.tensor(i_idx, dtype=torch.long, device=device)
            jt = torch.tensor(j_idx, dtype=torch.long, device=device)
            final_l, final_d = model(A_u, A_i, training=True, propagate=propagate)
            loss = bpr_loss(final_l, final_d, ut, it_, jt, REG_U) + weight_reg(model, REG_U)
            opt.zero_grad(); loss.backward(); opt.step()
    return model, A_u, A_i


@torch.no_grad()
def scores_for(model, A_u, A_i, u_idx, v_idx, device, propagate='e0', chunk=200000):
    model.eval()
    final_l, final_d = model(A_u, A_i, training=False, propagate=propagate)
    u = final_l[torch.tensor(u_idx, dtype=torch.long, device=device)]
    v = final_d[torch.tensor(v_idx, dtype=torch.long, device=device)]
    return (u * v).sum(1).cpu().numpy()


def threshold_metrics(y, s):
    out = {}
    pred = (s >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out['ACC@0.5'] = (tp + tn) / max(tp + tn + fp + fn, 1)
    # best-F1 threshold
    order = np.argsort(-s)
    ys = y[order]
    tp_c = np.cumsum(ys)
    fp_c = np.cumsum(1 - ys)
    f1 = 2 * tp_c / np.maximum(2 * tp_c + fp_c + (ys.sum() - tp_c), 1e-12)
    k = int(np.argmax(f1))
    thr = s[order][k]
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out['best_f1_thr'] = float(thr)
    out['ACC@bestF1'] = (tp + tn) / max(tp + tn + fp + fn, 1)
    out['F1@bestF1'] = 2 * tp / max(2 * tp + fp + fn, 1e-12)
    out['Precision@bestF1'] = tp / max(tp + fp, 1e-12)
    out['Recall@bestF1'] = tp / max(tp + fn, 1e-12)
    out['MCC@bestF1'] = ((tp * tn - fp * fn) /
                         np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-12)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=-1, help='-1 = all folds')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--propagate', choices=['e0', 'iter'], default='e0')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    n_l, n_d = D.full_sizes()
    ass = D.load_all()
    known = set(map(tuple, ass.tolist()))
    all_pairs = np.array([(i, j) for i in range(n_l) for j in range(n_d) if (i, j) not in known],
                         dtype=np.int64)

    os.makedirs(RESULTS, exist_ok=True)
    out = a.out or os.path.join(RESULTS, f'hgnnlda_seed{a.seed}.csv')
    folds = range(5) if a.fold < 0 else [a.fold]
    rows = []
    for f in folds:
        t0 = time.time()
        tr, te = D.load_fold(f)
        model, A_u, A_i = train_model(tr, n_l, n_d, a.device, a.seed + f, a.propagate)
        rng = np.random.default_rng(1000 + f)

        # balanced
        neg = D.sample_negatives(ass, n_l, n_d, len(te), rng)
        s_pos = scores_for(model, A_u, A_i, te[:, 0], te[:, 1], a.device, a.propagate)
        s_neg = scores_for(model, A_u, A_i, neg[:, 0], neg[:, 1], a.device, a.propagate)
        y = np.concatenate([np.ones(len(te)), np.zeros(len(neg))])
        s = np.concatenate([s_pos, s_neg])
        bal_auc = roc_auc_score(y, s); bal_aupr = average_precision_score(y, s)
        extra = threshold_metrics(y, s)

        # global
        sub = all_pairs if len(all_pairs) <= 200000 else all_pairs[rng.choice(len(all_pairs), 200000, replace=False)]
        s_all = scores_for(model, A_u, A_i, sub[:, 0], sub[:, 1], a.device, a.propagate)
        yg = np.concatenate([np.ones(len(te)), np.zeros(len(sub))])
        sg = np.concatenate([s_pos, s_all])
        g_auc = roc_auc_score(yg, sg); g_aupr = average_precision_score(yg, sg)

        row = dict(method='HGNNLDA', dataset='RNAactDrug', fold=f, seed=a.seed,
                   n_lncrna=n_l, n_drug=n_d, n_train=len(tr), n_test=len(te),
                   bal_auc=bal_auc, bal_aupr=bal_aupr,
                   glob_auc=g_auc, glob_aupr=g_aupr,
                   time_s=round(time.time() - t0, 1), **extra)
        rows.append(row)
        print(f'== fold {f}: balanced AUC {bal_auc:.4f} AUPR {bal_aupr:.4f} | '
              f'global AUC {g_auc:.4f} AUPR {g_aupr:.4f} | ACC@bestF1 {extra["ACC@bestF1"]:.4f} '
              f'({row["time_s"]}s)', flush=True)

    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('\n=== HGNNLDA RNAactDrug (mean over folds) ===')
    for k in ('bal_auc', 'bal_aupr', 'glob_auc', 'glob_aupr', 'ACC@bestF1', 'F1@bestF1', 'MCC@bestF1'):
        v = [r[k] for r in rows]
        print(f'  {k}: {st.mean(v):.4f} +/- {st.stdev(v) if len(v) > 1 else 0:.4f}')
    print('saved', out)


if __name__ == '__main__':
    main()
