# The full 8-cell grid: task x fold x metric, single model vs ensemble.
#
# "Stable" means positive in every cell, not significant in one.  Ensembling is
# the standard variance-reduction move and had only ever been applied to two of
# the eight cells here.  This measures base and posw2, each as a single model
# (10-seed mean) and as a 10-seed logit ensemble, on all eight.
#
# Each metric is read at its own validation-selected epoch, and the epoch is
# always chosen on VALIDATION -- for the test cells that is a clean held-out
# rule; for the validation cells it is the same rule the paper's own best-epoch
# table uses, and is labelled as such.
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.headroom import train_capture
from improve.run import TASKS

PAPER = {'LncDrug': dict(val=(0.9834, 0.9545), test=(0.9857, 0.9585)),
         'MiDrug': dict(val=(0.9555, 0.9183), test=(0.9564, 0.9165))}


def zs(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--tasks', nargs='+', default=list(TASKS))
    p.add_argument('--arms', nargs='+', default=['base', 'posw2'])
    p.add_argument('--seeds', nargs='+', type=int,
                   default=[666, 0, 1, 2, 3, 42, 7, 13, 21, 100])
    p.add_argument('--out', required=True)
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []

    for task in args.tasks:
        for arm in args.arms:
            acc = {f: {'auc': [], 'pr': [], 's_auc': [], 's_pr': []}
                   for f in ('val', 'test')}
            y = {}
            for seed in args.seeds:
                cap, _ = train_capture(args.run_dir, task, seed, device,
                                       arm=arm)
                for f in ('val', 'test'):
                    c = cap[f]
                    y[f] = c['y']
                    acc[f]['s_auc'].append(roc_auc_score(c['y'], c['score']))
                    acc[f]['s_pr'].append(
                        average_precision_score(c['y'], c['score_pr']))
                    acc[f]['auc'].append(zs(c['score']))
                    acc[f]['pr'].append(zs(c['score_pr']))
                print(f'{task} {arm} seed={seed} '
                      f"val {acc['val']['s_auc'][-1]:.4f}/"
                      f"{acc['val']['s_pr'][-1]:.4f} "
                      f"test {acc['test']['s_auc'][-1]:.4f}/"
                      f"{acc['test']['s_pr'][-1]:.4f}", flush=True)
            for f in ('val', 'test'):
                pa, pp = PAPER[task][f]
                rows.append(dict(
                    task=task, arm=arm, fold=f,
                    single_auc=float(np.mean(acc[f]['s_auc'])),
                    single_aupr=float(np.mean(acc[f]['s_pr'])),
                    ens_auc=roc_auc_score(y[f], np.mean(acc[f]['auc'], axis=0)),
                    ens_aupr=average_precision_score(
                        y[f], np.mean(acc[f]['pr'], axis=0)),
                    paper_auc=pa, paper_aupr=pp))
                pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print('\n=== 8-cell grid ===')
    print(df.to_string(index=False, float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    main()
