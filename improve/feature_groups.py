# How much of the published number is just node degree?
#
# `improve/altmodels.py` showed a tree on 19 explicit pair features beating the
# GNN.  The obvious objection is that the tree sees something the GNN does not:
# it reads the similarity matrices as CONTINUOUS values, while
# `heter_data_prepare.py` hands the GNN a version thresholded at 0.3.  This
# tests the objection by nesting the feature set, from the part the GNN
# unambiguously has to the part it does not:
#
#   degree  -- 5 features, endpoint and neighbourhood degree.  Nothing else.
#   graph   -- + path counts, SVD factors, the cross-ncRNA bridge.  Every one is
#              a function of edge sets the GNN also receives, so a win here
#              cannot be an information advantage.
#   all     -- + the continuous similarity-weighted collaborative scores, which
#              are the only features drawing on something the GNN gets in
#              binarised form.
#
# Two separate questions get answered at once: whether the tree's advantage is
# information (compare `graph` to `all`) and what the benchmark is actually
# measuring (compare `degree` to the paper).
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.altmodels import fitted
from improve.gbdt import build_inputs, featurise
from improve.run import TASKS

PAPER = {'LncDrug': dict(val=(0.9834, 0.9545), test=(0.9857, 0.9585)),
         'MiDrug': dict(val=(0.9555, 0.9183), test=(0.9564, 0.9165))}

DEGREE = ['log_deg_r', 'log_deg_d', 'pref_attach', 'nbr_deg_r', 'nbr_deg_d']
GRAPH = DEGREE + ['path3', 'path3_norm', 'path5', 'path5_norm', 'svd_dot',
                  'bridge', 'bridge_norm']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--net-dir', required=True)
    p.add_argument('--task', required=True, choices=list(TASKS))
    p.add_argument('--model', default='extratrees',
                   choices=['extratrees', 'gbdt', 'logreg'])
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1, 2, 3])
    p.add_argument('--out', required=True)
    args = p.parse_args()

    splits, key, A, S_r, S_d, bridge, attrs = build_inputs(
        args.run_dir, args.net_dir, args.task, 'cpu')
    rows = {n: (d[key].edge_label_index.cpu().numpy(),
                d[key].edge_label.cpu().numpy())
            for n, d in zip(('train', 'val', 'test'), splits)}
    X = {f: featurise(rows[f][0], A, S_r, S_d, bridge, Xr=attrs[0], Xd=attrs[1]) for f in ('val', 'test')}

    tr_idx, tr_y = rows['train']
    rng = np.random.RandomState(0)
    fold = rng.randint(0, args.folds, size=tr_idx.shape[1])
    parts = []
    for k in range(args.folds):
        m = fold == k
        Ak = A.copy()
        held = tr_idx[:, m & (tr_y == 1)]
        Ak[held[0], held[1]] = 0.0
        f = featurise(tr_idx[:, m], Ak, S_r, S_d, bridge, Xr=attrs[0], Xd=attrs[1])
        f.index = np.where(m)[0]
        parts.append(f)
    X['train'] = pd.concat(parts).sort_index()

    groups = [('degree', DEGREE), ('graph', GRAPH), ('all', list(X['train']))]
    res = []
    for name, cols in groups:
        acc = {f: {'auc': [], 'aupr': []} for f in ('val', 'test')}
        for seed in args.seeds:
            clf = fitted(args.model, seed)
            clf.fit(X['train'][cols].values, tr_y)
            for f in ('val', 'test'):
                s = clf.predict_proba(X[f][cols].values)[:, 1]
                acc[f]['auc'].append(roc_auc_score(rows[f][1], s))
                acc[f]['aupr'].append(
                    average_precision_score(rows[f][1], s))
        r = dict(task=args.task, model=args.model, group=name, n_feat=len(cols),
                 n_seeds=len(args.seeds))
        for f in ('val', 'test'):
            for m in ('auc', 'aupr'):
                r[f'{f}_{m}'] = float(np.mean(acc[f][m]))
                r[f'{f}_{m}_std'] = float(np.std(acc[f][m]))
            r[f'paper_{f}_auc'], r[f'paper_{f}_aupr'] = PAPER[args.task][f]
        res.append(r)
        print(f"  {name:8s} ({len(cols):2d} feat)  "
              f"val {r['val_auc']:.4f}/{r['val_aupr']:.4f}  "
              f"test {r['test_auc']:.4f}/{r['test_aupr']:.4f}", flush=True)

    df = pd.DataFrame(res)
    df.to_csv(args.out, index=False)
    pa, pp = PAPER[args.task]['val']
    d0 = df[df.group == 'degree'].iloc[0]
    print(f"\n{args.task}: paper S_val {pa:.4f}/{pp:.4f}  vs  "
          f"degree-only tree {d0['val_auc']:.4f}/{d0['val_aupr']:.4f}  "
          f"({d0['val_auc'] - pa:+.4f} / {d0['val_aupr'] - pp:+.4f})")


if __name__ == '__main__':
    main()
