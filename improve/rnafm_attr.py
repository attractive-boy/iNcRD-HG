# RNA-FM sequence embeddings as a two-task information channel.
#
# The released node attributes are LncExp / MiExp (a 938-dim expression profile)
# for the ncRNA side and ChemBERTa for the drug side.  A separate data build
# (`incrd-run/3.heter_data/*_rnafm.pth`) replaces the ncRNA side with 1578-dim
# RNA-FM sequence embeddings -- the same nodes, same protocol folds, different
# attribute channel.  This has never been used by `improve/`: every feature in
# `gbdt.py` reads the expression-profile attributes, and the GNN arms run on the
# released `.pth` files.  RNA-FM is symmetric across the two tasks, so unlike
# every mechanism since `posw2` it is not ipso facto single-task.
#
# The test: swap the ncRNA attribute block inside `featurise` from the released
# expression profile to the RNA-FM embedding, holding everything else fixed
# (same folds, same similarity matrices, same drug attributes, same members).
# The attr-pair features (`attr_cos_*`, `attr_maxcos_*`) then read the sequence
# channel.  If RNA-FM carries drug-degree-independent information, the members
# gain; if it is another view of the same popularity structure, they do not.
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.altmodels import fitted
from improve.gbdt import build_inputs, featurise
from improve.feature_extra import extra_columns
from improve.run import TASKS

PAPER = {'LncDrug': dict(val=(0.9834, 0.9545), test=(0.9857, 0.9585)),
         'MiDrug': dict(val=(0.9555, 0.9183), test=(0.9564, 0.9165))}
RNA = {'LncDrug': 'lncRNA', 'MiDrug': 'miRNA'}


def rnafm_attrs(rnafm_path, task, rna_type):
    d = torch.load(rnafm_path, map_location='cpu')
    return d[rna_type].x.numpy().astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--net-dir', required=True)
    p.add_argument('--rnafm', required=True,
                   help='dir holding {task}_train_data_rnafm.pth')
    p.add_argument('--task', required=True, choices=list(TASKS))
    p.add_argument('--model', default='extratrees',
                   choices=['extratrees', 'gbdt', 'logreg', 'mlp'])
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1])
    p.add_argument('--out', required=True)
    args = p.parse_args()

    splits, key, A, S_r, S_d, bridge, attrs = build_inputs(
        args.run_dir, args.net_dir, args.task, 'cpu')
    rna = RNA[args.task]
    rna_x = rnafm_attrs(f'{args.rnafm}/{args.task}_train_data_rnafm.pth',
                        args.task, rna)
    rna_attrs = (rna_x, attrs[1])          # swap ncRNA side, keep drug side

    rows = {n: (d[key].edge_label_index.cpu().numpy(),
                d[key].edge_label.cpu().numpy())
            for n, d in zip(('train', 'val', 'test'), splits)}
    y = {f: rows[f][1] for f in ('val', 'test')}

    def build(attr_pair, pairs, Amat):
        base = featurise(pairs, Amat, S_r, S_d, bridge,
                         Xr=attr_pair[0], Xd=attr_pair[1])
        return pd.concat([base.reset_index(drop=True),
                          extra_columns(pairs, Amat, S_r, S_d)], axis=1)

    # release attrs vs RNA-FM attrs, both on the SAME folds and matrix.
    tr_idx, tr_y = rows['train']
    rng = np.random.RandomState(0)
    fold = rng.randint(0, args.folds, size=tr_idx.shape[1])

    results = []
    for label, attr_pair in (('release_attr', attrs), ('rnafm_attr', rna_attrs)):
        Xtr_parts = []
        for k in range(args.folds):
            m = fold == k
            Ak = A.copy()
            held = tr_idx[:, m & (tr_y == 1)]
            Ak[held[0], held[1]] = 0.0
            f = build(attr_pair, tr_idx[:, m], Ak)
            f.index = np.where(m)[0]
            Xtr_parts.append(f)
        Xtr = pd.concat(Xtr_parts).sort_index()
        Xv = build(attr_pair, rows['val'][0], A)
        Xt = build(attr_pair, rows['test'][0], A)
        acc = {f: {'auc': [], 'aupr': []} for f in ('val', 'test')}
        for seed in args.seeds:
            clf = fitted(args.model, seed)
            clf.fit(Xtr.values, tr_y)
            for f, Xf in (('val', Xv), ('test', Xt)):
                s = clf.predict_proba(Xf.values)[:, 1]
                acc[f]['auc'].append(roc_auc_score(y[f], s))
                acc[f]['aupr'].append(average_precision_score(y[f], s))
        r = dict(task=args.task, model=args.model, attr=label)
        for f in ('val', 'test'):
            for m in ('auc', 'aupr'):
                r[f'{f}_{m}'] = float(np.mean(acc[f][m]))
        results.append(r)
        print(f"  {label:13s} val {r['val_auc']:.4f}/{r['val_aupr']:.4f}  "
              f"test {r['test_auc']:.4f}/{r['test_aupr']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    pa = PAPER[args.task]
    print(f'\n{args.task} {args.model}: paper S_val '
          f'{pa["val"][0]:.4f}/{pa["val"][1]:.4f}  '
          f'test {pa["test"][0]:.4f}/{pa["test"][1]:.4f}')


if __name__ == '__main__':
    main()