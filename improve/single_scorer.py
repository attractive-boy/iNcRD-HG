# One scoring model for BOTH datasets, and it must beat the per-dataset
# rank-fusion (LncDrug 0.9874/0.9714, MiDrug 0.9582/0.9225 on S_val).
#
# The fusion wins by mixing model-class diversity: GNN (graph structure),
# trees/linear (explicit pair features), NCN (common-neighbour counts).  A
# single GBDT cannot be three classes at once, but it CAN eat all three
# channels as features -- including the GNN's own score -- and learn a better
# combination than a per-dataset linear rank average.  That is one model, one
# hyperparameter vector, used identically for both ncRNA types.
#
# Leakage guard, the same shape as gbdt.py/gnn_emb_feat.py:
#   * val/test pair features come from the full training association matrix.
#   * train pair features are K-fold: fold k featurised from a matrix with
#     fold k's positives removed.
#   * val/test GNN score comes from a GNN trained on the full training graph.
#   * train GNN score is OOF: fold k scored by a GNN trained on the graph with
#     fold k's positives removed.
#
# --combined trains ONE model on LncDrug-train + MiDrug-train rows pooled and
# scores both validation folds with that single model (truly "one model for
# both datasets").  Without it, the same architecture/hyperparams are trained
# per dataset, which is "one model" in the method sense.
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.feature_extra import extra_columns
from improve.gbdt import build_inputs, featurise
from improve.run import ARMS, TASKS, load_splits, set_seed

FUSION_VAL = {'LncDrug': (0.9874, 0.9714), 'MiDrug': (0.9582, 0.9225)}


def rk(v):
    return v.argsort().argsort().astype(np.float64) / max(len(v) - 1, 1)


def train_scores(train_data, task, device, key, pairs, epochs=None, lr=None):
    """Train the author's base GNN on `train_data` (already fold-guarded) and
    return the best-epoch logits on `pairs` (the fold's own rows)."""
    import torch.nn.functional as F
    import torch_geometric.utils as geo_utils
    from improve.model import Model

    t = TASKS[task]
    set_seed(666)
    model = Model(train_data, t['channels'], t['aggr'], t['rna']).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=lr or t['lr'], alpha=0.99)
    label_index = train_data[key].edge_label_index
    label = train_data[key].edge_label
    n_src = train_data[t['rna']].num_nodes
    n_dst = train_data['drug'].num_nodes
    best = (0.0, None)
    pidx = torch.as_tensor(pairs, device=device)
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
            v_auc = roc_auc_score(label.cpu().numpy(), pv.cpu().numpy())
        if v_auc > best[0]:
            best = (v_auc, ep)
            with torch.no_grad():
                s = model(train_data.x_dict, train_data.edge_index_dict,
                          pidx).cpu().numpy()
    return best[1], s


def task_features(args, task, device, gnn_scores):
    """Return (X, y) for train/val/test plus per-fold train GNN scores."""
    t = TASKS[task]
    key = (t['rna'], task, 'drug')
    splits, _, A, S_r, S_d, bridge, attrs = build_inputs(
        args.run_dir, args.net_dir, task, 'cpu')
    rows = {n: (d[key].edge_label_index.cpu().numpy(),
                d[key].edge_label.cpu().numpy())
            for n, d in zip(('train', 'val', 'test'), splits)}
    y = {f: rows[f][1] for f in ('val', 'test')}

    base = {f: featurise(rows[f][0], A, S_r, S_d, bridge,
                         Xr=attrs[0], Xd=attrs[1])
            for f in ('val', 'test')}
    X = {f: pd.concat([
        base[f].reset_index(drop=True),
        extra_columns(rows[f][0], A, S_r, S_d),
        pd.DataFrame({f'gs_{f}': gnn_scores[f]}),
    ], axis=1) for f in ('val', 'test')}

    tr_idx, tr_y = rows['train']
    rng = np.random.RandomState(0)
    fold = rng.randint(0, args.folds, size=tr_idx.shape[1])
    parts = []
    for k in range(args.folds):
        m = fold == k
        train_data = load_splits(args.run_dir, task, ARMS['base'], device)[0]
        pos_idx = train_data[key].edge_label_index[:, tr_y == 1]
        fold_pos = tr_idx[:, m & (tr_y == 1)]
        ei = train_data[key].edge_index.cpu().numpy()
        rem = set(map(tuple, fold_pos.T.tolist()))
        keep = np.array([tuple(p) not in rem for p in ei.T])
        if not keep.all():
            train_data[key].edge_index = torch.tensor(ei[:, keep], device=device)
            keep_l = np.array([tuple(p) not in rem
                               for p in train_data[key].edge_label_index
                               .cpu().numpy().T])
            train_data[key].edge_label_index = train_data[key] \
                .edge_label_index[:, keep_l]
            train_data[key].edge_label = train_data[key].edge_label[keep_l]
        _, s_fold = train_scores(train_data, task, device, key,
                                 tr_idx[:, m])
        Ak = A.copy()
        held = tr_idx[:, m & (tr_y == 1)]
        Ak[held[0], held[1]] = 0.0
        f = featurise(tr_idx[:, m], Ak, S_r, S_d, bridge,
                      Xr=attrs[0], Xd=attrs[1])
        f = pd.concat([f.reset_index(drop=True),
                       extra_columns(tr_idx[:, m], Ak, S_r, S_d),
                       pd.DataFrame({'gs_train': rk(s_fold)})], axis=1)
        f.index = np.where(m)[0]
        parts.append(f)
    X['train'] = pd.concat(parts).sort_index()
    return X, y, tr_y


def fit_eval(model, X, y, tr_y):
    model.fit(X['train'].values, tr_y)
    for f in ('val', 'test'):
        s = model.predict_proba(X[f].values)[:, 1]
        print(f'    {f:4s} AUC {roc_auc_score(y[f], s):.4f}  '
              f'AUPR {average_precision_score(y[f], s):.4f}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tasks = ['LncDrug', 'MiDrug']
    data = {}
    for task in tasks:
        tag = 'lnc' if task == 'LncDrug' else 'mi'
        a = np.load(os.path.join(ARGS.fusion_dir, f'alt_{tag}_seed666.npz'))
        gnn = {'val': a['gnn_val'], 'test': a['gnn_test']}
        print(f'== {task}: building features + OOF GNN scores', flush=True)
        X, y, tr_y = task_features(ARGS, task, device, gnn)
        data[task] = (X, y, tr_y)
        pa, pp = FUSION_VAL[task]
        print(f'   fusion target val {pa:.4f}/{pp:.4f}', flush=True)

    def scorer():
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=False, random_state=666)

    if not ARGS.combined:
        for task in tasks:
            X, y, tr_y = data[task]
            print(f'== {task}: single scorer (per-dataset training, same arch/hp)')
            fit_eval(scorer(), X, y, tr_y)
    else:
        Xc = {f: pd.concat([data[t][0][f] for t in tasks]).reset_index(drop=True)
              for f in ('train', 'val', 'test')}
        tr_y = np.concatenate([data[t][2] for t in tasks])
        print('== COMBINED: one model trained on both datasets')
        m = scorer()
        m.fit(Xc['train'].values, tr_y)
        for task in tasks:
            idx = np.arange(len(data[task][1]['val']))
            s = m.predict_proba(Xc['val'].values[idx])[:, 1]
            print(f'    {task} val AUC {roc_auc_score(data[task][1]["val"], s):.4f}'
                  f'  AUPR {average_precision_score(data[task][1]["val"], s):.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--net-dir', required=True)
    p.add_argument('--fusion-dir', required=True)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--combined', action='store_true')
    ARGS = p.parse_args()
    main()