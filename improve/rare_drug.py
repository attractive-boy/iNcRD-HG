# Localise the cross-drug residual, then attack it.
#
# LncDrug's error is 0.0117 and all of it is in cross-drug pairs.  A cross-drug
# pair is ranked wrong when a positive on drug A loses to a negative on drug B,
# which under a degree-dominated scorer happens when A is rarer than B.  So the
# residual should be concentrated on positives sitting on low-degree drugs.
#
# Part 1 measures that: per-positive win rate against all negatives, against
# the drug's training degree, and how much of the total error the bottom half
# of the degree range accounts for.
#
# Part 2 tests the fix that follows from it.  The raw degree prior gives a rare
# drug a low score with no way to distinguish "rare and unlike anything known"
# from "rare but chemically close to well-studied drugs".  A similarity-smoothed
# propensity does distinguish them:
#
#     smoothed(d) = (deg(d) + sum_{d' ~ d} S(d,d') * deg(d')) / (1 + sum S)
#
# over the DrugDrug similarity edges already in the graph.  This is the one
# place where real drug chemistry can move the released metric, because it is
# the only place the metric still has error.
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.headroom import train_capture, zscore
from improve.run import TASKS


def drug_similarity(run_dir, n_drug):
    """Dense drug-drug similarity from the released matrix, diagonal removed."""
    S = pd.read_csv(f'{run_dir}/2.construct_network/drugSimMat.csv',
                    header=None).values.astype(float)
    S = np.abs(S)[:n_drug, :n_drug]
    np.fill_diagonal(S, 0.0)
    return S


def smoothed_degree(deg, S, thresh, power):
    """Similarity-weighted neighbour degree, added to a drug's own degree.

    `thresh` keeps the smoothing local (the graph's own similarity edges use
    0.3); `power` sharpens the weights so a near-identical drug counts for much
    more than a marginally similar one.
    """
    W = np.where(S > thresh, S ** power, 0.0)
    num = deg + W @ deg
    den = 1.0 + W.sum(axis=1)
    return num / np.maximum(den, 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--tasks', nargs='+', default=list(TASKS))
    p.add_argument('--seeds', nargs='+', type=int,
                   default=[666, 0, 1, 2, 3, 42, 7, 13, 21, 100])
    p.add_argument('--arm', default='base')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rows, loc = [], []
    for task in args.tasks:
        t = TASKS[task]
        key = (t['rna'], task, 'drug')
        train = torch.load(f'{args.run_dir}/3.heter_data/{task}_train_data.pth',
                           map_location='cpu')
        n_drug = train['drug'].num_nodes
        S = drug_similarity(args.run_dir, n_drug)
        ei = train[key].edge_index
        deg = torch.zeros(n_drug)
        deg.scatter_add_(0, ei[1], torch.ones_like(ei[1], dtype=torch.float))
        deg = deg.numpy()

        # Grid for the smoothing, selected on validation only.
        grid = [(th, pw, a) for th in (0.2, 0.3, 0.5)
                for pw in (1, 2, 4) for a in (0.1, 0.2, 0.4, 0.8)]

        for seed in args.seeds:
            cap, _ = train_capture(args.run_dir, task, seed, device,
                                   arm=args.arm)
            y, s, dr = cap['test']['y'], cap['test']['score'], cap['test']['drug']

            # --- Part 1: where is the residual? ---
            pos, neg = s[y == 1], s[y == 0]
            win = (pos[:, None] > neg[None, :]).mean(axis=1)   # per-positive AUC
            pdeg = deg[dr[y == 1]]
            med = np.median(deg[deg > 0])
            lo, hi = pdeg <= med, pdeg > med
            # Error mass = sum of (1 - win), i.e. how many pairs each group loses.
            err = 1.0 - win
            loc.append(dict(
                task=task, seed=seed,
                win_lowdeg=win[lo].mean() if lo.any() else np.nan,
                win_highdeg=win[hi].mean() if hi.any() else np.nan,
                n_lowdeg=int(lo.sum()), n_highdeg=int(hi.sum()),
                err_share_lowdeg=err[lo].sum() / err.sum() if err.sum() else np.nan,
                pos_share_lowdeg=lo.mean(),
                spearman_win_deg=float(pd.Series(win).corr(pd.Series(pdeg),
                                                           method='spearman'))))

            # --- Part 2: similarity-smoothed propensity, alpha+grid on val ---
            best, best_va = None, -1.0
            zv, zt = zscore(cap['val']['score']), zscore(s)
            for th, pw, a in grid:
                sm = smoothed_degree(deg, S, th, pw)
                fv = zscore(np.log1p(sm[cap['val']['drug']]))
                va = roc_auc_score(cap['val']['y'], zv + a * fv)
                if va > best_va:
                    best_va, best = va, (th, pw, a)
            th, pw, a = best
            sm = smoothed_degree(deg, S, th, pw)
            ft = zscore(np.log1p(sm[dr]))
            blend = zt + a * ft
            # Raw-degree blend at its own val-selected alpha, as the control the
            # smoothing has to beat -- otherwise this is just the old prior again.
            best_raw, best_raw_va = 0.0, -1.0
            rv = zscore(np.log1p(deg[cap['val']['drug']]))
            for a2 in (0.0, 0.1, 0.2, 0.4, 0.8):
                va = roc_auc_score(cap['val']['y'], zv + a2 * rv)
                if va > best_raw_va:
                    best_raw_va, best_raw = va, a2
            raw_blend = zt + best_raw * zscore(np.log1p(deg[dr]))

            rows.append(dict(
                task=task, seed=seed, thresh=th, power=pw, alpha=a,
                model=roc_auc_score(y, s), model_aupr=average_precision_score(y, s),
                smooth_prior_only=roc_auc_score(y, ft),
                raw_prior_only=roc_auc_score(y, np.log1p(deg[dr])),
                blend_smooth=roc_auc_score(y, blend),
                blend_smooth_aupr=average_precision_score(y, blend),
                blend_raw=roc_auc_score(y, raw_blend)))
            print(f"{task} seed={seed} model={rows[-1]['model']:.4f} "
                  f"smooth_only={rows[-1]['smooth_prior_only']:.4f} "
                  f"raw_only={rows[-1]['raw_prior_only']:.4f} "
                  f"blend_smooth={rows[-1]['blend_smooth']:.4f} "
                  f"blend_raw={rows[-1]['blend_raw']:.4f} "
                  f"(th={th},pw={pw},a={a})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    pd.DataFrame(loc).to_csv(args.out.replace('.csv', '_localise.csv'),
                             index=False)
    print('\n=== Part 1: where the residual sits ===')
    print(pd.DataFrame(loc).groupby('task')[
        ['win_lowdeg', 'win_highdeg', 'pos_share_lowdeg', 'err_share_lowdeg',
         'spearman_win_deg']].mean().to_string(float_format=lambda v: f'{v:.4f}'))
    print('\n=== Part 2: similarity-smoothed propensity ===')
    df['d_vs_model'] = df['blend_smooth'] - df['model']
    df['d_vs_rawblend'] = df['blend_smooth'] - df['blend_raw']
    print(df.groupby('task')[['model', 'raw_prior_only', 'smooth_prior_only',
                              'blend_raw', 'blend_smooth', 'd_vs_model',
                              'd_vs_rawblend']].mean()
          .to_string(float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    main()
