# Cross-task propensity: use the OTHER ncRNA task's degree for rare drugs.
#
# The residual on LncDrug is 21.5% carried by 2.6% of positives, all sitting on
# drugs that are rare in lncRNA space (per-positive win rate 0.889 vs 0.989).
# Chemical-similarity smoothing of the degree does not fix them
# (improve/rare_drug.py: the smoothed prior is WORSE than the raw one).
#
# But `RandomLinkSplit` only splits the target relation, so when LncDrug is the
# task the full MiDrug edge set is present in the graph -- and a drug that is
# rare in lncRNA space may be well studied in miRNA space.  That cross-task
# degree is a propensity signal the model has never been given explicitly, and
# it is the paper's own pan-ncRNA premise stated as a prior.
#
# Arms, all fitted on the training fold's labels and all selected on validation:
#   own        log1p(degree in the target relation)                  [the old control]
#   cross      + log1p(degree in the other ncRNA relation)
#   chem       + a ChemBERTa-driven propensity (ridge on drug features)
#   full       own + cross + chem + ncRNA degree
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.headroom import train_capture, zscore
from improve.run import TASKS

OTHER = {'LncDrug': ('MiDrug', 'miRNA'), 'MiDrug': ('LncDrug', 'lncRNA')}


def build_features(run_dir, task):
    """Per-(ncRNA, drug) propensity features, from the training graph only."""
    t = TASKS[task]
    key = (t['rna'], task, 'drug')
    train = torch.load(f'{run_dir}/3.heter_data/{task}_train_data.pth',
                       map_location='cpu')
    n_drug = train['drug'].num_nodes

    ei = train[key].edge_index
    own = torch.zeros(n_drug)
    own.scatter_add_(0, ei[1], torch.ones_like(ei[1], dtype=torch.float))
    rna_deg = torch.zeros(train[t['rna']].num_nodes)
    rna_deg.scatter_add_(0, ei[0], torch.ones_like(ei[0], dtype=torch.float))

    # The other task's relation is untouched by this task's split, so its full
    # edge set is in the graph the model already message-passes over.  Reading
    # it here adds no information the model did not have.
    o_task, o_rna = OTHER[task]
    o_ei = train[o_rna, o_task, 'drug'].edge_index
    cross = torch.zeros(n_drug)
    cross.scatter_add_(0, o_ei[1], torch.ones_like(o_ei[1], dtype=torch.float))

    # ChemBERTa -> expected degree.  A learned map, where rare_drug.py's
    # fingerprint-similarity smoothing was a fixed kernel.
    dx = train['drug'].x.cpu().numpy()
    chem = Ridge(alpha=10.0).fit(dx, np.log1p(own.numpy())).predict(dx)

    return dict(own=own.numpy(), cross=cross.numpy(), chem=chem,
                rna_deg=rna_deg.numpy(),
                tr_idx=train[key].edge_label_index.numpy(),
                tr_y=train[key].edge_label.numpy())


def cols(f, rna_idx, drug_idx, which):
    out = []
    if 'own' in which:
        out.append(np.log1p(f['own'][drug_idx]))
    if 'cross' in which:
        out.append(np.log1p(f['cross'][drug_idx]))
    if 'chem' in which:
        out.append(f['chem'][drug_idx])
    if 'rna' in which:
        out.append(np.log1p(f['rna_deg'][rna_idx]))
    return np.stack(out, axis=1)


ARMS = {'own': ('own',), 'cross': ('own', 'cross'),
        'chem': ('own', 'chem'), 'full': ('own', 'cross', 'chem', 'rna')}


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
    rows = []

    for task in args.tasks:
        f = build_features(args.run_dir, task)
        t = TASKS[task]
        key = (t['rna'], task, 'drug')
        fitted = {}
        for name, which in ARMS.items():
            X = cols(f, f['tr_idx'][0], f['tr_idx'][1], which)
            fitted[name] = LogisticRegression(max_iter=2000).fit(X, f['tr_y'])

        rna_idx = {}
        for fold in ('val', 'test'):
            d = torch.load(f'{args.run_dir}/3.heter_data/{task}_{fold}_data.pth',
                           map_location='cpu')
            rna_idx[fold] = d[key].edge_label_index[0].numpy()

        for seed in args.seeds:
            cap, _ = train_capture(args.run_dir, task, seed, device, arm=args.arm)
            y, s = cap['test']['y'], cap['test']['score']
            row = dict(task=task, seed=seed, model=roc_auc_score(y, s),
                       model_aupr=average_precision_score(y, s))
            zv, zt = zscore(cap['val']['score']), zscore(s)
            alphas = np.concatenate([[0.0], np.linspace(0.05, 2.0, 40)])

            for name, which in ARMS.items():
                pri = {}
                for fold in ('val', 'test'):
                    X = cols(f, rna_idx[fold], cap[fold]['drug'], which)
                    pri[fold] = fitted[name].decision_function(X)
                row[f'prior_{name}'] = roc_auc_score(y, pri['test'])
                pv, pt = zscore(pri['val']), zscore(pri['test'])
                va = [roc_auc_score(cap['val']['y'], zv + a * pv) for a in alphas]
                a_star = float(alphas[int(np.argmax(va))])
                row[f'blend_{name}'] = roc_auc_score(y, zt + a_star * pt)
                row[f'blend_{name}_aupr'] = average_precision_score(
                    y, zt + a_star * pt)
                row[f'alpha_{name}'] = a_star
            rows.append(row)
            print(f"{task} seed={seed} model={row['model']:.4f} | prior "
                  + ' '.join(f"{n}={row[f'prior_{n}']:.4f}" for n in ARMS)
                  + ' | blend '
                  + ' '.join(f"{n}={row[f'blend_{n}']:.4f}" for n in ARMS),
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print('\n=== priors alone (test AUC) ===')
    print(df.groupby('task')[[f'prior_{n}' for n in ARMS]].mean()
          .to_string(float_format=lambda v: f'{v:.4f}'))
    print('\n=== blended into the model ===')
    for n in ARMS:
        df[f'd_{n}'] = df[f'blend_{n}'] - df['model']
    print(df.groupby('task')[['model'] + [f'blend_{n}' for n in ARMS]
                             + [f'd_{n}' for n in ARMS]].mean()
          .to_string(float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    main()
