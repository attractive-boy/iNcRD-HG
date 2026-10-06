# -*- coding: utf-8 -*-
"""HGNNLDA independent-dataset evaluation (train on all known associations)."""
import argparse
import csv
import os

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

import hgnnlda_data as D
from run_hgnnlda import train_model, scores_for, threshold_metrics, RESULTS

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--device', default='cuda' if __import__('torch').cuda.is_available() else 'cpu')
    a = ap.parse_args()

    n_l, n_d = D.full_sizes()
    ass = D.load_all()
    ind = D.load_independent()
    model, A_u, A_i = train_model(ass, n_l, n_d, a.device, a.seed)

    rng = np.random.default_rng(7)
    neg = D.sample_negatives(ass, n_l, n_d, len(ind), rng)
    s_pos = scores_for(model, A_u, A_i, ind[:, 0], ind[:, 1], a.device)
    s_neg = scores_for(model, A_u, A_i, neg[:, 0], neg[:, 1], a.device)
    y = np.concatenate([np.ones(len(ind)), np.zeros(len(neg))])
    s = np.concatenate([s_pos, s_neg])
    row = dict(method='HGNNLDA', dataset='RNAactDrug-independent', seed=a.seed,
               n_test=len(ind), bal_auc=roc_auc_score(y, s),
               bal_aupr=average_precision_score(y, s), **threshold_metrics(y, s))
    print('independent:', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f'hgnnlda_independent_seed{a.seed}.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys())); w.writeheader(); w.writerow(row)
    print('saved', out)


if __name__ == '__main__':
    main()
