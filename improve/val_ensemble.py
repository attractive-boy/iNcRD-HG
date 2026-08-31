# Ensemble on the validation fold, to match the paper's ablation table.
#
# The paper's S_val row prints one number per cell.  A 10-seed mean is a
# different quantity from a single printed number, so this reports three things
# side by side and labels which is which:
#
#   mean      -- 10-seed mean +- std, the conservative figure
#   best seed -- the maximum over seeds, which is what a single printed number
#                may well be, but is a selection over 10 runs
#   ensemble  -- logits averaged over the 10 seeds at each model's own
#                best-validation epoch, then scored once.  This is ONE model in
#                the sense that matters: it is a fixed, reproducible predictor,
#                not a pick of the luckiest run.
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from improve.headroom import train_capture
from improve.run import TASKS

PAPER = {'LncDrug': (0.9834, 0.9545), 'MiDrug': (0.9555, 0.9183)}


def zs(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--tasks', nargs='+', default=list(TASKS))
    p.add_argument('--arm', default='posw2')
    p.add_argument('--seeds', nargs='+', type=int,
                   default=[666, 0, 1, 2, 3, 42, 7, 13, 21, 100])
    p.add_argument('--out', required=True)
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []

    for task in args.tasks:
        vs, vps, singles = [], [], []
        for seed in args.seeds:
            cap, _ = train_capture(args.run_dir, task, seed, device,
                                   arm=args.arm)
            v, vp, yv = (cap['val']['score'], cap['val']['score_pr'],
                         cap['val']['y'])
            vs.append(zs(v)); vps.append(zs(vp))
            singles.append((roc_auc_score(yv, v),
                            average_precision_score(yv, vp)))
            print(f'{task} seed={seed} val_auc={singles[-1][0]:.4f} '
                  f'val_aupr={singles[-1][1]:.4f}', flush=True)
        singles = np.array(singles)
        ens = np.mean(vs, axis=0); ens_pr = np.mean(vps, axis=0)
        pa, pp = PAPER[task]
        rows.append(dict(
            task=task, arm=args.arm, n=len(vs),
            mean_auc=singles[:, 0].mean(), std_auc=singles[:, 0].std(),
            best_auc=singles[:, 0].max(),
            ens_auc=roc_auc_score(yv, ens),
            mean_aupr=singles[:, 1].mean(), std_aupr=singles[:, 1].std(),
            best_aupr=singles[:, 1].max(),
            ens_aupr=average_precision_score(yv, ens_pr),
            paper_auc=pa, paper_aupr=pp))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    for _, r in df.iterrows():
        print(f"\n{r['task']}  (paper {r['paper_auc']:.4f} / {r['paper_aupr']:.4f})")
        for lab, a, u in (('10-seed mean', r['mean_auc'], r['mean_aupr']),
                          ('best seed   ', r['best_auc'], r['best_aupr']),
                          ('ensemble    ', r['ens_auc'], r['ens_aupr'])):
            print(f"  {lab}  AUC {a:.4f} {'OK ' if a > r['paper_auc'] else '   '}"
                  f" AUPR {u:.4f} {'OK' if u > r['paper_aupr'] else ''}")


if __name__ == '__main__':
    main()
