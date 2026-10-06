"""Author-protocol GBDT member with the ncRNA-disease x drug-disease channel.

Reuses improve.gbdt featurisation and 5-fold leakage guard, appends the label-free
disease bridge columns, and dumps val/test scores in the layout blend.py expects.

Run:
  /data1/licheng/incrd-hg/.venv-dgl/bin/python \
      improve/research/ncrna_disease_next_20260917/author_disease_gbdt_v1.py \
      --run-dir /data1/licheng/incrd-hg/incrd-orig --task MiDrug \
      --mode disease|control|base --out /path/gb_dis_MiDrug.npz
"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG')
from improve import gbdt as G  # noqa: E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    'dav', '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/ncrna_disease_next_20260917/disease_anchor_verify_v1.py')
dav = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dav)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--task', required=True, choices=['MiDrug', 'LncDrug'])
    p.add_argument('--mode', default='disease', choices=['base', 'disease', 'control'])
    p.add_argument('--perm-seed', type=int, default=20260917)
    p.add_argument('--disease-cols', default='all', choices=['all', 'pair'])
    p.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1, 2, 3])
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    net_dir = args.run_dir + '/2.construct_network'
    splits, key, A, S_r, S_d, bridge, attrs = G.build_inputs(args.run_dir, net_dir, args.task, 'cpu')
    train, val, test = splits
    rows = {n: (d[key].edge_label_index.cpu().numpy(), d[key].edge_label.cpu().numpy())
            for n, d in (('train', train), ('val', val), ('test', test))}

    X = {n: G.featurise(rows[n][0], A, S_r, S_d, bridge, Xr=attrs[0], Xd=attrs[1])
         for n in ('val', 'test')}
    tr_idx, tr_y = rows['train']
    rng = np.random.RandomState(0)
    fold = rng.randint(0, args.folds, size=tr_idx.shape[1])
    parts = []
    for k in range(args.folds):
        m = fold == k
        Ak = A.copy()
        held = tr_idx[:, m & (tr_y == 1)]
        Ak[held[0], held[1]] = 0.0
        f = G.featurise(tr_idx[:, m], Ak, S_r, S_d, bridge, Xr=attrs[0], Xd=attrs[1])
        f.index = np.where(m)[0]
        parts.append(f)
    X['train'] = pd.concat(parts).sort_index()

    if args.mode != 'base':
        ch = dav.Channel(permute=(args.mode == 'control'), seed=args.perm_seed)
        D = {'train': ch.features(args.task, tr_idx),
             'val': ch.features(args.task, rows['val'][0]),
             'test': ch.features(args.task, rows['test'][0])}
        if args.disease_cols == 'pair':
            D = {n: v[:, :3] for n, v in D.items()}
        for n in ('train', 'val', 'test'):
            X[n] = pd.DataFrame(np.c_[X[n].values, D[n]])

    scores = {f: [] for f in ('val', 'test')}
    for seed in args.seeds:
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                             l2_regularization=1.0, early_stopping=False, random_state=seed)
        clf.fit(X['train'].values, tr_y)
        for f in ('val', 'test'):
            s = clf.predict_proba(X[f].values)[:, 1]
            scores[f].append(s)
            print('seed %-4d %s auc %.4f aupr %.4f' % (seed, f, roc_auc_score(rows[f][1], s),
                                                       average_precision_score(rows[f][1], s)), flush=True)
    np.savez(args.out, **{f: np.mean(scores[f], axis=0) for f in ('val', 'test')},
             y_val=rows['val'][1], y_test=rows['test'][1])
    print('wrote', args.out)


if __name__ == '__main__':
    main()
