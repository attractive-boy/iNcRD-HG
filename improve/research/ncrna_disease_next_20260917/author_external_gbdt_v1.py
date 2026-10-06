"""Author-protocol GBDT member with selectable external information channels."""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG')
from improve import gbdt as G  # noqa: E402
import importlib.util  # noqa: E402

HERE = '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/ncrna_disease_next_20260917'
spec = importlib.util.spec_from_file_location('dav', HERE + '/disease_anchor_verify_v1.py')
dav = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dav)
spec2 = importlib.util.spec_from_file_location('ec', HERE + '/external_channels_v1.py')
ec = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(ec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--task', required=True, choices=['MiDrug', 'LncDrug'])
    p.add_argument('--channels', default='disease', help='comma list: disease,perturb')
    p.add_argument('--mode', default='real', choices=['real', 'control', 'none'])
    p.add_argument('--perm-seed', type=int, default=20260917)
    p.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1])
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
    fold = rng.randint(0, 5, size=tr_idx.shape[1])
    parts = []
    for k in range(5):
        m = fold == k
        Ak = A.copy()
        held = tr_idx[:, m & (tr_y == 1)]
        Ak[held[0], held[1]] = 0.0
        f = G.featurise(tr_idx[:, m], Ak, S_r, S_d, bridge, Xr=attrs[0], Xd=attrs[1])
        f.index = np.where(m)[0]
        parts.append(f)
    X['train'] = pd.concat(parts).sort_index()

    if args.mode != 'none':
        chan = args.channels.split(',')
        blocks = []
        if 'disease' in chan:
            ch = dav.Channel(permute=(args.mode == 'control'), seed=args.perm_seed)
            blocks.append({n: ch.features(args.task, rows[n][0]) for n in ('train', 'val', 'test')})
        if 'diseasesim' in chan:
            spec3 = importlib.util.spec_from_file_location('dsim', HERE + '/diseasesim_channel_v1.py')
            ds = importlib.util.module_from_spec(spec3)
            spec3.loader.exec_module(ds)
            dsc = ds.DiseaseSimChannel(permute=(args.mode == 'control'), seed=args.perm_seed)
            blocks.append({n: dsc.features(args.task, rows[n][0]) for n in ('train', 'val', 'test')})
        if 'mechanism' in chan:
            dch = dav.Channel(permute=False)
            mc = ec.MechanismChannel(dch, permute=(args.mode == 'control'), seed=args.perm_seed)
            blocks.append({n: mc.features(args.task, rows[n][0]) for n in ('train', 'val', 'test')})
        if 'perturb' in chan:
            pc = ec.PerturbChannel(permute=(args.mode == 'control'), seed=args.perm_seed)
            blocks.append({n: pc.features(rows[n][0]) for n in ('train', 'val', 'test')})
        for n in ('train', 'val', 'test'):
            extra = np.c_[tuple(b[n] for b in blocks)]
            keep = np.isfinite(extra).all(axis=1)
            extra = np.nan_to_num(extra, nan=0.0)
            X[n] = pd.DataFrame(np.c_[X[n].values, extra])
            if n == 'train':
                tr_keep = keep
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
