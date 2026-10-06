# -*- coding: utf-8 -*-
"""DAESTB -- leak-free variant: GIP similarity re-computed per fold from the TRAINING
associations only, autoencoder fit on training features only, features for the test
pairs derived from those training-only similarities.  This is the fair (inductive)
counterpart of daestb.py (which uses the full association matrix, transductive).
"""
import argparse
import csv
import os
import statistics as st
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEV, RESULTS, LOGS, load_fold, seed_it, compute_metrics
from daestb import DeepAE, gip_kernel, AE_EPOCHS, AE_BATCH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    seed_it(args.seed)
    device = DEV

    fold_data = {f: load_fold(f) for f in range(5)}
    all_pairs = []
    for f in range(5):
        all_pairs += fold_data[f][0] + fold_data[f][2]
    all_pairs = list(dict.fromkeys(all_pairs))
    drugs = list(dict.fromkeys(s for s, _ in all_pairs))
    mirnas = list(dict.fromkeys(q for _, q in all_pairs))
    d_idx = {s: i for i, s in enumerate(drugs)}
    m_idx = {q: i for i, q in enumerate(mirnas)}

    def feats(pairs, DS, MS):
        H = np.empty((len(pairs), DS.shape[1] + MS.shape[1]), dtype=np.float32)
        for r, (s, q) in enumerate(pairs):
            H[r, :DS.shape[1]] = DS[d_idx[s]]
            H[r, DS.shape[1]:] = MS[m_idx[q]]
        return H

    from xgboost import XGBClassifier
    fields = ['dataset', 'method', 'fold', 'seed', 'auc', 'aupr', 'acc', 'precision',
              'recall', 'specificity', 'f1', 'mcc', 'best_epoch', 'last_auc', 'last_aupr']
    rows = []
    tag = os.environ.get('DLST_TAG', '')
    with open(os.path.join(LOGS, f'daestb_foldwise{tag}_seed{args.seed}.log'), 'w') as log:
        for f in range(5):
            tr_pairs, tr_y, te_pairs, te_y = fold_data[f]
            A = np.zeros((len(mirnas), len(drugs)), dtype=np.float32)
            for (s, q), y in zip(tr_pairs, tr_y):
                A[m_idx[q], d_idx[s]] = int(y)
            DS = gip_kernel(A.T)
            MS = gip_kernel(A)
            Htr = feats(tr_pairs, DS, MS)
            Hte = feats(te_pairs, DS, MS)
            X = torch.tensor(Htr, device=device)
            ae = DeepAE(X.shape[1]).to(device)
            opt = torch.optim.Adam(ae.parameters())
            lossf = nn.MSELoss()
            for _ in range(AE_EPOCHS):
                ae.train()
                perm = torch.randperm(X.size(0), device=device)
                for i in range(0, X.size(0), AE_BATCH):
                    b = X[perm[i:i + AE_BATCH]]
                    opt.zero_grad()
                    rec, _ = ae(b)
                    lossf(rec, b).backward()
                    opt.step()
            ae.eval()
            with torch.no_grad():
                Ztr = ae.enc(torch.tensor(Htr, device=device)).cpu().numpy()
                Zte = ae.enc(torch.tensor(Hte, device=device)).cpu().numpy()
            clf = XGBClassifier(n_estimators=100, learning_rate=0.3, eval_metric='logloss',
                                verbosity=0)
            clf.fit(Ztr, np.asarray(tr_y).astype(int))
            p = clf.predict_proba(Zte)[:, 1]
            m = compute_metrics(np.asarray(te_y).astype(int), p)
            rows.append({'dataset': 'DLST-MDA-1:1', 'method': 'DAESTB(leak-free)', 'fold': f,
                         'seed': args.seed, 'auc': m['AUC'], 'aupr': m['AUPR'], 'acc': m['ACC'],
                         'precision': m['Precision'], 'recall': m['Recall'],
                         'specificity': m['Specificity'], 'f1': m['F1'], 'mcc': m['MCC'],
                         'best_epoch': 0, 'last_auc': m['AUC'], 'last_aupr': m['AUPR']})
            msg = f'== daestb_leakfree fold={f} AUC={m["AUC"]:.4f} AUPR={m["AUPR"]:.4f}'
            print(msg, flush=True)
            log.write(msg + '\n')
    out = args.out or os.path.join(RESULTS, 'daestb_foldwise_per_fold.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print('\n=== DAESTB leak-free ===')
    for k in ['auc', 'aupr', 'acc', 'f1', 'mcc']:
        v = [r[k] for r in rows]
        print(f'  {k}: {st.mean(v):.4f} +/- {st.stdev(v):.4f}')
    print('saved', out)


if __name__ == '__main__':
    main()
