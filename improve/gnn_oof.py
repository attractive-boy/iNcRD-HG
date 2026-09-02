# GNN out-of-fold scores, so the GNN can enter the training-fold-fitted stack.
#
# `improve/stack.py` fits its meta-learner on the training fold's out-of-fold
# member scores, but the GNN is the one member that cannot produce them -- its
# training-fold scores would mean retraining it K times, so it enters only as a
# validation-selected scalar.  This script closes that gap: for fold k it trains
# a base GNN on the training graph with fold k's positives removed (the same
# per-fold guard as `gnn_emb_feat.py`), scores fold k's rows, and repeats.  A
# full-graph GNN scores validation and test.  The result is a `gnn` member with
# honest out-of-fold training scores, usable by `stack.py --gnn-oof`.
#
# The GNN here is the author's `base` config, NOT the `multiens` uniform
# ensemble that `stack.py --gnn` reads.  That is deliberate: OOF train scores
# and val/test scores must come from the same model class, and retraining the
# multi-config ensemble K times is not affordable.  The base GNN is weaker than
# the ensemble (val ~0.984 vs 0.9846) but internally consistent, which is what
# the stack needs.
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.gbdt import build_inputs
from improve.run import ARMS, TASKS, load_splits, set_seed

FOLDS = ('train', 'val', 'test')


def train_predictor(train_data, task, key, device, epochs=None):
    """Train a base GNN on the given (possibly edge-pruned) training object and
    return it, for scoring arbitrary edge pairs."""
    import torch_geometric.utils as geo_utils
    from improve.model import Model

    t = TASKS[task]
    set_seed(666)
    model = Model(train_data, t['channels'], t['aggr'], t['rna']).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=t['lr'], alpha=0.99)
    label_index = train_data[key].edge_label_index
    label = train_data[key].edge_label
    n_src = train_data[t['rna']].num_nodes
    n_dst = train_data['drug'].num_nodes
    best = (0.0, model.state_dict())
    for ep in range(1, epochs or t['epochs']):
        model.train()
        opt.zero_grad()
        pos = label_index[:, label == 1]
        if pos.size(1) == 0:
            continue
        neg = geo_utils.negative_sampling(
            pos, num_nodes=(n_src, n_dst), num_neg_samples=2 * pos.size(1),
            method='sparse')
        idx = torch.cat([pos, neg], dim=1)
        lbl = torch.cat([torch.ones(pos.size(1), device=device),
                         torch.zeros(neg.size(1), device=device)])
        pred = model(train_data.x_dict, train_data.edge_index_dict, idx)
        loss = F.binary_cross_entropy_with_logits(pred, lbl)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(train_data.x_dict, train_data.edge_index_dict,
                       label_index)
            a = roc_auc_score(label.cpu().numpy(), pv.cpu().numpy())
        if a > best[0]:
            best = (a, {k: v.detach().cpu().clone() for k, v in
                        model.state_dict().items()})
    model.load_state_dict(best[1])
    return model


def score(model, data, key, idx, device):
    model.eval()
    with torch.no_grad():
        s = model(data.x_dict, data.edge_index_dict, idx).cpu().numpy()
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--task', required=True, choices=list(TASKS))
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--epochs', type=int, default=None,
                   help='override the author epochs for a quicker run')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = TASKS[args.task]
    key = (t['rna'], args.task, 'drug')

    splits, _, A, _, _, _, _ = build_inputs(args.run_dir,
                                            args.run_dir + '/2.construct_network',
                                            args.task, 'cpu')
    rows = {n: (d[key].edge_label_index.cpu().numpy(),
                d[key].edge_label.cpu().numpy())
            for n, d in zip(FOLDS, splits)}
    tr_idx, tr_y = rows['train']
    rng = np.random.RandomState(0)
    fold = rng.randint(0, args.folds, size=tr_idx.shape[1])

    oof = np.zeros(tr_idx.shape[1])
    for k in range(args.folds):
        m = fold == k
        train_data = load_splits(args.run_dir, args.task, ARMS['base'],
                                 device)[0]
        ei = train_data[key].edge_index.cpu().numpy()
        fold_pos = tr_idx[:, m & (tr_y == 1)]
        rem = set(map(tuple, fold_pos.T.tolist()))
        keep = np.array([tuple(p) not in rem for p in ei.T])
        if not keep.all():
            train_data[key].edge_index = torch.tensor(ei[:, keep],
                                                      device=device)
            keep_l = np.array([tuple(p) not in rem for p in
                               train_data[key].edge_label_index
                               .cpu().numpy().T])
            train_data[key].edge_label_index = train_data[key] \
                .edge_label_index[:, keep_l]
            train_data[key].edge_label = train_data[key].edge_label[keep_l]
        model = train_predictor(train_data, args.task, key, device,
                                epochs=args.epochs)
        oof[m] = score(model, train_data, key,
                       torch.tensor(tr_idx[:, m], device=device), device)
        print(f'  fold {k}: {m.sum()} rows, oof AUC '
              f'{roc_auc_score(tr_y[m], oof[m]):.4f}', flush=True)

    # full-graph GNN for val/test
    train_data = load_splits(args.run_dir, args.task, ARMS['base'], device)[0]
    model = train_predictor(train_data, args.task, key, device,
                            epochs=args.epochs)
    out = {'gnn_train': oof, 'y_train': tr_y}
    for f in ('val', 'test'):
        out[f'gnn_{f}'] = score(model, train_data, key,
                                torch.tensor(rows[f][0], device=device),
                                device)
        out[f'y_{f}'] = rows[f][1]
    np.savez(args.out, **out)
    print(f'wrote {args.out}: gnn_train oof AUC '
          f'{roc_auc_score(tr_y, oof):.4f}')


if __name__ == '__main__':
    main()