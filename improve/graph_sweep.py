# Sweep the similarity-graph construction, which the released method never did.
#
# `improve/graph_prep.py` shows the split is invariant to these edges, so every
# variant here is scored on exactly the released validation fold -- same
# positives, same negatives, same held-out target edges.  The only thing that
# moves is the graph the encoder passes messages over, and S_val is computed on
# the TRAINING graph, so this is a lever on validation that does not touch a
# label.
#
# Two families:
#   thr  -- the released rule, |sim| > t, but with a per-relation t instead of
#           one global 0.3
#   knn  -- each node keeps its k most similar partners, so degree is uniform
#           across nodes and comparable across the three relations
import argparse
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.final import capture
from improve.graph_prep import build, seed_all, split_all
from improve.run import TASKS

RELS = [('lncRNA', 'LncLnc', 'lncRNA'), ('miRNA', 'MiMi', 'miRNA'),
        ('drug', 'DrugDrug', 'drug')]


def make_splits(net_dir, spec, device):
    seed_all()
    data = build(net_dir, spec['lnc'], spec['mi'], spec['drug'],
                 knn=spec['knn'])
    deg = {r[1]: data[r].edge_index.size(1) / data[r[0]].num_nodes
           for r in RELS}
    return split_all(data, device), deg


def thin_train(splits, task, rna, frac, seed=0):
    """Keep `frac` of the training positives, message edges and supervision.

    A mechanism control, not a protocol: the validation and test folds are
    untouched, only the density of the association graph the encoder sees.
    MiDrug carries 8.9 training drug edges per miRNA against LncDrug's 0.96,
    which is the standing explanation for why the similarity graph is worth
    tuning on one task and not the other.  Thinning MiDrug to LncDrug's density
    tests it directly.
    """
    if frac >= 1.0:
        return splits
    train = splits[0]
    key = (rna, task, 'drug')
    rev = ('drug', 'rev_' + task, rna)
    g = torch.Generator().manual_seed(seed)

    ei = train[key].edge_index
    keep = torch.randperm(ei.size(1), generator=g)[:int(round(frac * ei.size(1)))]
    keep, _ = keep.sort()
    train[key].edge_index = ei[:, keep.to(ei.device)]
    if rev in train.edge_types:
        train[rev].edge_index = train[key].edge_index.flip(0)

    # Supervision: keep the same positives, and the protocol's 2:1 negatives.
    li, lb = train[key].edge_label_index, train[key].edge_label
    pos = torch.where(lb == 1)[0]
    neg = torch.where(lb == 0)[0]
    kp = pos[keep.to(pos.device)]
    kn = neg[torch.randperm(neg.numel(), generator=g)[:2 * kp.numel()].to(
        neg.device)]
    sel = torch.cat([kp, kn])
    train[key].edge_label_index = li[:, sel]
    train[key].edge_label = lb[sel]
    return splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--net-dir', required=True)
    p.add_argument('--task', required=True, choices=list(TASKS))
    p.add_argument('--arm', default='posw2')
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--channels', type=int, required=True)
    p.add_argument('--aggr', required=True)
    p.add_argument('--epochs', type=int, default=301)
    p.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1])
    p.add_argument('--grid', required=True,
                   help='semicolon-separated specs, each "lnc,mi,drug[,knn]" '
                        'e.g. "0.3,0.3,0.3; 0.3,0.3,0.2; 10,10,10,knn"')
    p.add_argument('--thin-train', type=float, default=1.0,
                   help='keep this fraction of training positives (mechanism '
                        'control; val/test folds untouched)')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    specs = []
    for chunk in args.grid.split(';'):
        parts = [c.strip() for c in chunk.strip().split(',') if c.strip()]
        if not parts:
            continue
        knn = len(parts) > 3 and parts[3] == 'knn'
        specs.append(dict(lnc=parts[0], mi=parts[1], drug=parts[2], knn=knn))

    rows, t0 = [], time.time()
    for i, spec in enumerate(specs, 1):
        splits_all, deg = make_splits(args.net_dir, spec, device)
        splits = [d.to(device) for d in splits_all[args.task]]
        splits = thin_train(splits, args.task, TASKS[args.task]['rna'],
                            args.thin_train)
        n_tr = splits[0][(TASKS[args.task]['rna'], args.task,
                          'drug')].edge_index.size(1)
        acc = {f: {'auc': [], 'aupr': []} for f in ('val', 'test')}
        y = None
        for seed in args.seeds:
            cap, y = capture(splits, args.task, args.arm, args.lr,
                             args.channels, args.aggr, args.epochs, seed,
                             device)
            for f in ('val', 'test'):
                acc[f]['auc'].append(roc_auc_score(y[f], cap['auc'][f]))
                acc[f]['aupr'].append(
                    average_precision_score(y[f], cap['aupr'][f]))
        r = dict(task=args.task, arm=args.arm, lr=args.lr,
                 channels=args.channels, aggr=args.aggr,
                 family='knn' if spec['knn'] else 'thr',
                 lnc=spec['lnc'], mi=spec['mi'], drug=spec['drug'],
                 n_seeds=len(args.seeds), thin=args.thin_train, n_train=n_tr,
                 deg_lnc=deg['LncLnc'], deg_mi=deg['MiMi'],
                 deg_drug=deg['DrugDrug'])
        for f in ('val', 'test'):
            for m in ('auc', 'aupr'):
                r[f'{f}_{m}'] = float(np.mean(acc[f][m]))
                r[f'{f}_{m}_std'] = float(np.std(acc[f][m]))
        rows.append(r)
        el = (time.time() - t0) / 60
        print(f"[{i}/{len(specs)}] {r['family']} "
              f"lnc={spec['lnc']} mi={spec['mi']} drug={spec['drug']} "
              f"| deg {deg['LncLnc']:.1f}/{deg['MiMi']:.1f}/"
              f"{deg['DrugDrug']:.1f} "
              f"| val {r['val_auc']:.4f}/{r['val_aupr']:.4f} "
              f"test {r['test_auc']:.4f}/{r['test_aupr']:.4f} "
              f"[{el:.1f}m, eta {el/i*(len(specs)-i):.0f}m]", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        del splits, splits_all
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values('val_auc', ascending=False)
    df.to_csv(args.out, index=False)
    print('\n=== sorted by val AUC ===')
    print(df[['family', 'lnc', 'mi', 'drug', 'deg_drug', 'val_auc', 'val_aupr',
              'test_auc', 'test_aupr']].to_string(
                  index=False, float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    main()
