# The ceiling of the propensity family, measured with an oracle that cheats.
#
# Five propensity sources have failed on LncDrug (own degree, two-sided degree,
# chemical-similarity smoothing, cross-task degree, ChemBERTa regression).  That
# says the estimators failed; it does not say the information is absent.  This
# separates the two.
#
# The oracle reads the TEST fold's own labels to compute each drug's and each
# ncRNA's empirical positive rate, then scores test edges with them.  It cannot
# be built without the answers, so it is not a method -- it is an upper bound on
# every scorer whose only inputs are the endpoints' identities.  Anything a
# better propensity estimator could ever reach lies below it.
#
# Also reported: the ceiling once the model is blended with the oracle, which
# bounds "model + perfect propensity" -- the best possible version of every
# blend arm in this repo.
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


def empirical_rate(ids, y, n, prior_w=1.0):
    """Leave-one-out P(y=1 | endpoint), estimated on the fold it scores.

    The edge's own label MUST be excluded from its own endpoint's rate.  With
    it included, an endpoint appearing once in the fold gets a rate equal to
    its label, and the "oracle" measures label memorisation instead of
    propensity -- which inflates it towards 1.0 and makes the bound useless.
    """
    pos = np.zeros(n)
    tot = np.zeros(n)
    np.add.at(pos, ids, y)
    np.add.at(tot, ids, 1.0)
    base = y.mean()
    num = pos[ids] - y + prior_w * base
    den = tot[ids] - 1.0 + prior_w
    return num / den


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--tasks', nargs='+', default=list(TASKS))
    p.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1, 2, 3])
    p.add_argument('--arm', default='base')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []

    for task in args.tasks:
        t = TASKS[task]
        key = (t['rna'], task, 'drug')
        test = torch.load(f'{args.run_dir}/3.heter_data/{task}_test_data.pth',
                          map_location='cpu')
        rna_t = test[key].edge_label_index[0].numpy()
        n_rna = test[t['rna']].num_nodes
        n_drug = test['drug'].num_nodes

        for seed in args.seeds:
            cap, _ = train_capture(args.run_dir, task, seed, device,
                                   arm=args.arm)
            y, s, drug_t = cap['test']['y'], cap['test']['score'], cap['test']['drug']

            # Oracles: the endpoint's true positive rate in this very fold.
            od = empirical_rate(drug_t, y, n_drug)
            orr = empirical_rate(rna_t, y, n_rna)
            X = np.stack([od, orr], axis=1)
            ob = LogisticRegression(max_iter=1000).fit(X, y).decision_function(X)

            row = dict(task=task, seed=seed, model=roc_auc_score(y, s),
                       model_aupr=average_precision_score(y, s),
                       oracle_drug=roc_auc_score(y, od),
                       oracle_rna=roc_auc_score(y, orr),
                       oracle_both=roc_auc_score(y, ob),
                       oracle_both_aupr=average_precision_score(y, ob))
            # Best blend of the model with the oracle, alpha chosen ON TEST --
            # deliberately optimistic, since this is an upper bound.  AUC and
            # AUPR are maximised independently: AUPR is not a function of AUC,
            # so its ceiling has to be measured, not inferred from the AUC one.
            zt, zo = zscore(s), zscore(ob)
            grid = [zt + a * zo for a in np.linspace(0.0, 4.0, 81)]
            row['blend_oracle_best'] = max(roc_auc_score(y, g) for g in grid)
            row['blend_oracle_best_aupr'] = max(
                average_precision_score(y, g) for g in grid)
            rows.append(row)
            print(f"{task} seed={seed} model={row['model']:.4f} "
                  f"oracle_drug={row['oracle_drug']:.4f} "
                  f"oracle_rna={row['oracle_rna']:.4f} "
                  f"oracle_both={row['oracle_both']:.4f} "
                  f"model+oracle={row['blend_oracle_best']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    df['headroom'] = df['blend_oracle_best'] - df['model']
    df['headroom_aupr'] = df['blend_oracle_best_aupr'] - df['model_aupr']
    print('\n=== upper bound on the whole propensity family ===')
    print(df.groupby('task')[['model', 'oracle_drug', 'oracle_rna',
                              'oracle_both', 'blend_oracle_best', 'headroom']]
          .mean().to_string(float_format=lambda v: f'{v:.4f}'))
    print('\n=== the same bound, measured on AUPR ===')
    print(df.groupby('task')[['model_aupr', 'oracle_both_aupr',
                              'blend_oracle_best_aupr', 'headroom_aupr']]
          .mean().to_string(float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    main()
