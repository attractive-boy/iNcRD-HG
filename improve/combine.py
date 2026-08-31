# Last combination: ensemble + the full propensity prior, on LncDrug.
#
# Sixteen arms have been measured on LncDrug and exactly two are nominally
# positive, neither significant on its own: averaging logits over seeds
# (+0.0002) and blending all four propensity features (+0.0002).  They are
# independent -- one reduces init variance, the other adds information the
# model does not have -- so they should stack.  This is the last untested
# combination, not a new idea.
#
# Reported against BOTH baselines it has to beat:
#   single  -- the mean single-seed model, i.e. what the paper reports
#   ens     -- the ensemble alone, so the prior's contribution is separable
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.cross_task_prior import ARMS as PRIOR_ARMS, build_features, cols
from improve.headroom import train_capture, zscore
from improve.run import TASKS
from sklearn.linear_model import LogisticRegression


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--task', default='LncDrug')
    p.add_argument('--arm', default='base')
    p.add_argument('--seeds', nargs='+', type=int,
                   default=[666, 0, 1, 2, 3, 42, 7, 13, 21, 100])
    p.add_argument('--boot', type=int, default=2000,
                   help='bootstrap resamples over test edges for the CI')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task = args.task
    t = TASKS[task]
    key = (t['rna'], task, 'drug')

    f = build_features(args.run_dir, task)
    which = PRIOR_ARMS['full']
    X_tr = cols(f, f['tr_idx'][0], f['tr_idx'][1], which)
    clf = LogisticRegression(max_iter=2000).fit(X_tr, f['tr_y'])

    rna_idx = {}
    for fold in ('val', 'test'):
        d = torch.load(f'{args.run_dir}/3.heter_data/{task}_{fold}_data.pth',
                       map_location='cpu')
        rna_idx[fold] = d[key].edge_label_index[0].numpy()

    val_s, test_s, singles = [], [], []
    for seed in args.seeds:
        cap, _ = train_capture(args.run_dir, task, seed, device, arm=args.arm)
        val_s.append(zscore(cap['val']['score']))
        test_s.append(zscore(cap['test']['score']))
        y = cap['test']['y']
        singles.append(roc_auc_score(y, cap['test']['score']))
        print(f'{task} seed={seed} single={singles[-1]:.4f}', flush=True)
    yv, y = cap['val']['y'], cap['test']['y']
    drug_v, drug_t = cap['val']['drug'], cap['test']['drug']

    pri_v = zscore(clf.decision_function(cols(f, rna_idx['val'], drug_v, which)))
    pri_t = zscore(clf.decision_function(cols(f, rna_idx['test'], drug_t, which)))

    ens_v, ens_t = np.mean(val_s, axis=0), np.mean(test_s, axis=0)
    alphas = np.concatenate([[0.0], np.linspace(0.02, 1.5, 75)])
    a = float(alphas[int(np.argmax(
        [roc_auc_score(yv, zscore(ens_v) + al * pri_v) for al in alphas]))])
    comb_t = zscore(ens_t) + a * pri_t

    res = dict(
        single_mean=float(np.mean(singles)), single_std=float(np.std(singles)),
        ens=roc_auc_score(y, ens_t), comb=roc_auc_score(y, comb_t), alpha=a,
        single_mean_aupr=float(np.mean(
            [average_precision_score(y, s) for s in test_s])),
        ens_aupr=average_precision_score(y, ens_t),
        comb_aupr=average_precision_score(y, comb_t))

    # Bootstrap over test edges: the ensemble is one scoring function, so a
    # seed-paired test does not apply.  Resample edges instead, keeping the
    # positive/negative counts fixed.
    rng = np.random.default_rng(0)
    pos_i, neg_i = np.where(y == 1)[0], np.where(y == 0)[0]
    d_ens, d_comb, d_pri = [], [], []
    mean_single = np.mean(test_s, axis=0)  # for reference only
    for _ in range(args.boot):
        bp = rng.choice(pos_i, len(pos_i), replace=True)
        bn = rng.choice(neg_i, len(neg_i), replace=True)
        idx = np.concatenate([bp, bn])
        yy = y[idx]
        base_b = np.mean([roc_auc_score(yy, s[idx]) for s in test_s])
        d_ens.append(roc_auc_score(yy, ens_t[idx]) - base_b)
        d_comb.append(roc_auc_score(yy, comb_t[idx]) - base_b)
        d_pri.append(roc_auc_score(yy, comb_t[idx])
                     - roc_auc_score(yy, ens_t[idx]))
    for name, v in (('ens_vs_single', d_ens), ('comb_vs_single', d_comb),
                    ('comb_vs_ens', d_pri)):
        v = np.array(v)
        res[f'{name}_delta'] = float(v.mean())
        res[f'{name}_lo'] = float(np.percentile(v, 2.5))
        res[f'{name}_hi'] = float(np.percentile(v, 97.5))
        res[f'{name}_pgt0'] = float((v > 0).mean())

    pd.DataFrame([res]).to_csv(args.out, index=False)
    print('\n' + '\n'.join(f'{k:22s} {v:.5f}' if isinstance(v, float) else
                           f'{k:22s} {v}' for k, v in res.items()))


if __name__ == '__main__':
    main()
