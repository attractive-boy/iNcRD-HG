# Paired bootstrap between two score dumps, over fold edges.
#
# `improve/stack.py` reports one number per configuration averaged over seeds,
# and RESULTS.md has already established that the binding uncertainty on these
# folds is not seed spread but the fold itself: 1263 validation rows (421
# positive) on LncDrug give a bootstrap sd of 0.002-0.006 on AUC and AUPR.  So a
# difference between two configurations has to be resampled over edges, paired
# on the same resample, or it cannot be claimed.
#
# Positives and negatives are resampled separately with their counts held
# fixed, which keeps the class balance of every resample equal to the fold's --
# the same construction `improve/combine.py` uses.
import argparse

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

SCORER = {'auc': roc_auc_score, 'aupr': average_precision_score}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a', required=True, help='npz for the baseline arm')
    p.add_argument('--b', required=True, help='npz for the arm being tested')
    p.add_argument('--key', required=True,
                   help="score prefix inside both npz, e.g. val_grid or "
                        "stackgnn_nnls")
    p.add_argument('--label', default=None)
    p.add_argument('--boot', type=int, default=2000)
    args = p.parse_args()

    A, B = np.load(args.a), np.load(args.b)
    lab = args.label or f'{args.key}: b - a'
    print(f'{lab}   {args.boot} resamples\n')
    for fold in ('val', 'test'):
        y = A[f'y_{fold}']
        assert np.array_equal(y, B[f'y_{fold}']), f'{fold} label order differs'
        sa, sb = A[f'{args.key}_{fold}'], B[f'{args.key}_{fold}']
        rng = np.random.default_rng(0)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        idx = [np.concatenate([rng.choice(pos, len(pos), replace=True),
                               rng.choice(neg, len(neg), replace=True)])
               for _ in range(args.boot)]
        for metric in ('auc', 'aupr'):
            sc = SCORER[metric]
            base, new = sc(y, sa), sc(y, sb)
            d = np.array([sc(y[i], sb[i]) - sc(y[i], sa[i]) for i in idx])
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(f'  {fold:4s} {metric.upper():4s}  {base:.4f} -> {new:.4f}   '
                  f'delta {new - base:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   '
                  f'P(delta>0) = {(d > 0).mean():.3f}')
        print()


if __name__ == '__main__':
    main()
