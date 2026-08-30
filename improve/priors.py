# Is there error left in the cross-drug pairs, and can a two-sided prior take it?
#
# The same-drug / cross-drug pair decomposition bounds what a better WITHIN-drug
# ranker can win (0.0012).  It says nothing about the cross-drug pairs, which
# are 99.6% of the metric and where the model is at 0.9866, not 1.0.  Two
# questions follow:
#
# 1. Decompose the model's own AUC into its same-drug and cross-drug parts.
#    Global AUC is a weighted average of the two, so this says exactly which
#    half the remaining error is in.
# 2. RandomLinkSplit samples negatives uniformly over (ncRNA, drug), so
#    positives concentrate on popular ncRNAs as well as popular drugs.  Only
#    the drug side has ever been tested here.  Fit a prior on BOTH degrees and
#    see whether it beats the drug-only 0.9775 -- and whether blending it into
#    the model beats the model.
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.headroom import train_capture, zscore
from improve.run import TASKS


def auc_decomposition(y, score, drug):
    """Split P(s_pos > s_neg) into the same-drug and cross-drug pair sets.

    Global AUC is exactly the pair-count-weighted average of the two, so this
    localises the residual error rather than merely describing it.
    """
    pos, neg = score[y == 1], score[y == 0]
    dpos, dneg = drug[y == 1], drug[y == 0]
    # (n_pos, n_neg) comparison matrix; both folds are small enough.
    wins = (pos[:, None] > neg[None, :]).astype(float)
    wins += 0.5 * (pos[:, None] == neg[None, :])
    same = dpos[:, None] == dneg[None, :]
    out = {'auc_global': wins.mean(), 'n_same': int(same.sum()),
           'n_cross': int((~same).sum())}
    out['auc_same'] = wins[same].mean() if same.any() else np.nan
    out['auc_cross'] = wins[~same].mean()
    return out


def degrees(run_dir, task):
    """Drug and ncRNA training degree, from training message-passing edges."""
    t = TASKS[task]
    d = torch.load(f'{run_dir}/3.heter_data/{task}_train_data.pth',
                   map_location='cpu')
    ei = d[t['rna'], task, 'drug'].edge_index
    drug_deg = torch.zeros(d['drug'].num_nodes)
    drug_deg.scatter_add_(0, ei[1], torch.ones_like(ei[1], dtype=torch.float))
    rna_deg = torch.zeros(d[t['rna']].num_nodes)
    rna_deg.scatter_add_(0, ei[0], torch.ones_like(ei[0], dtype=torch.float))
    # Training labels, for fitting the prior.  Train labels only -- val is left
    # for selection and test is never touched.
    e = d[t['rna'], task, 'drug']
    return (drug_deg.numpy(), rna_deg.numpy(),
            e.edge_label_index.numpy(), e.edge_label.numpy())


def feats(rna_idx, drug_idx, rna_deg, drug_deg):
    return np.stack([np.log1p(drug_deg[drug_idx]),
                     np.log1p(rna_deg[rna_idx])], axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--tasks', nargs='+', default=list(TASKS))
    p.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1, 2, 3])
    p.add_argument('--arm', default='base',
                   help='model arm to blend on top of; neg resampling was the '
                        'one arm that helped MiDrug, so the two may stack')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows, decomp = [], []

    for task in args.tasks:
        drug_deg, rna_deg, tr_idx, tr_y = degrees(args.run_dir, task)
        # Priors fitted on the training fold's own labels.
        X_tr = feats(tr_idx[0], tr_idx[1], rna_deg, drug_deg)
        lr_both = LogisticRegression(max_iter=1000).fit(X_tr, tr_y)
        lr_rna = LogisticRegression(max_iter=1000).fit(X_tr[:, 1:], tr_y)

        for seed in args.seeds:
            cap, _ = train_capture(args.run_dir, task, seed, device, arm=args.arm)
            t = TASKS[task]
            d_test = torch.load(f'{args.run_dir}/3.heter_data/{task}_test_data.pth',
                                map_location='cpu')
            key = (t['rna'], task, 'drug')
            rna_idx = {k: None for k in cap}
            for k, fold in (('val', 'val'), ('test', 'test')):
                dd = torch.load(
                    f'{args.run_dir}/3.heter_data/{task}_{fold}_data.pth',
                    map_location='cpu')
                rna_idx[k] = dd[key].edge_label_index[0].numpy()

            pri = {}
            for k in ('val', 'test'):
                X = feats(rna_idx[k], cap[k]['drug'], rna_deg, drug_deg)
                pri[k] = dict(
                    drug=np.log1p(drug_deg[cap[k]['drug']]),
                    rna=lr_rna.decision_function(X[:, 1:]),
                    both=lr_both.decision_function(X))

            y, s = cap['test']['y'], cap['test']['score']
            row = dict(task=task, seed=seed, model=roc_auc_score(y, s))
            for name in ('drug', 'rna', 'both'):
                row[f'prior_{name}'] = roc_auc_score(y, pri['test'][name])

            # Blend the model with the two-sided prior; alpha chosen on val.
            alphas = np.concatenate([[0.0], np.linspace(0.05, 4.0, 80)])
            zv, zt = zscore(cap['val']['score']), zscore(s)
            pv, pt = zscore(pri['val']['both']), zscore(pri['test']['both'])
            va = [roc_auc_score(cap['val']['y'], zv + a * pv) for a in alphas]
            a_star = float(alphas[int(np.argmax(va))])
            row['alpha'] = a_star
            row['blend_both'] = roc_auc_score(y, zt + a_star * pt)
            row['blend_both_aupr'] = average_precision_score(y, zt + a_star * pt)
            row['model_aupr'] = average_precision_score(y, s)
            rows.append(row)
            print(f"{task} seed={seed} model={row['model']:.4f} "
                  f"prior_drug={row['prior_drug']:.4f} "
                  f"prior_rna={row['prior_rna']:.4f} "
                  f"prior_both={row['prior_both']:.4f} "
                  f"blend={row['blend_both']:.4f} (a={a_star:.2f})", flush=True)

            dc = auc_decomposition(y, s, cap['test']['drug'])
            dc.update(task=task, seed=seed, which='model')
            decomp.append(dc)
            dc2 = auc_decomposition(y, pri['test']['both'], cap['test']['drug'])
            dc2.update(task=task, seed=seed, which='prior_both')
            decomp.append(dc2)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    pd.DataFrame(decomp).to_csv(args.out.replace('.csv', '_decomp.csv'),
                                index=False)
    df['d_blend'] = df['blend_both'] - df['model']
    print('\n=== priors and blend (test AUC) ===')
    print(df.groupby('task')[['model', 'prior_drug', 'prior_rna', 'prior_both',
                              'blend_both', 'd_blend', 'alpha']].mean()
          .to_string(float_format=lambda v: f'{v:.4f}'))
    print('\n=== AUC decomposition: where is the error? ===')
    print(pd.DataFrame(decomp).groupby(['task', 'which'])
          [['auc_global', 'auc_same', 'auc_cross']].mean()
          .to_string(float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    main()
