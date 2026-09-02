# GNN embeddings as input features to the tree/MLP pool members.
#
# The reverse of `improve/hybrid.py`, which concatenated the 19 pair features
# INTO the GNN decoder and failed (2x256 embedding dims swamp the side path).
# This instead feeds the GNN's own learned embeddings INTO the tree member, as
# explicit numbers, so the tree sees both the hand-designed path counts and the
# latent structure the GNN recovered.  Score-level fusion can only mix ranks
# linearly; here the tree can learn interactions between explicit features and
# latent embeddings directly.
#
# Leakage is the whole difficulty again, and it is harder than in gbdt.py.
# A node embedding is a function of the whole message-passing graph, so a
# training positive (r,d) leaks into z_r and z_d through the edge itself.  The
# guard is per-fold GNN training: for fold k, a GNN is trained on the graph
# with fold k's positives removed, and that fold's rows get embeddings from it.
# Validation and test rows use the full-training-graph GNN (their edges are not
# in it).  This is 5 GNN trainings per task at the author's base config; each
# is ~101 epochs on a graph with < 1.5k ncRNA nodes, so it is cheap.
#
# Features derived from the pair of embeddings (all functions of z_r and z_d):
#
#   emb_dot     -- z_r . z_d, the bilinear term the decoder's first layer sees.
#   emb_norm    -- |z_r - z_d|_2, a distance in the latent space.
#   emb_cos     -- cosine similarity of the two embeddings.
#   emb_dot_a   -- z_r . z_d after each is standardised (removes scale).
#   emb_head_k  -- the leading principal components of the CONCATENATED pair,
#                  so the tree gets the highest-variance latent directions as
#                  plain numbers without dumping 2x256 dims in.
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
from improve.run import ARMS, TASKS, load_splits, run_one, set_seed
from improve.feature_extra import extra_columns

PAPER = {'LncDrug': dict(val=(0.9834, 0.9545), test=(0.9857, 0.9585)),
         'MiDrug': dict(val=(0.9555, 0.9183), test=(0.9564, 0.9165))}


def train_embeddings(run_dir, task, device, epochs=None, lr=None):
    """Train the author's base GNN and return val-epoch embeddings."""
    import torch.nn.functional as F
    import torch_geometric.utils as geo_utils
    from improve.model import Model

    t = TASKS[task]
    cfg = ARMS['base']
    key = (t['rna'], task, 'drug')

    set_seed(666)
    train_data = load_splits(run_dir, task, cfg, device)[0]
    val_data = load_splits(run_dir, task, cfg, device)[1]
    model = Model(train_data, t['channels'], t['aggr'], t['rna']).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=lr or t['lr'], alpha=0.99)
    label_index = train_data[key].edge_label_index
    label = train_data[key].edge_label
    n_src = train_data[t['rna']].num_nodes
    n_dst = train_data['drug'].num_nodes
    best = (0.0, None)
    for ep in range(1, epochs or t['epochs']):
        model.train()
        opt.zero_grad()
        pos = label_index[:, label == 1]
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
                       val_data[key].edge_label_index)
            v_auc = roc_auc_score(val_data[key].edge_label.cpu().numpy(),
                                  pv.cpu().numpy())
        if v_auc > best[0]:
            best = (v_auc, ep)
            with torch.no_grad():
                z_full = model.encoder(train_data.x_dict,
                                       train_data.edge_index_dict)
    return z_full, best


def train_embeddings_from_data(train_data, task, device, key, epochs=None,
                               lr=None):
    """Like train_embeddings but on an already-modified training object, and
    scoring on the training set itself (there is no held-out val in a fold)."""
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
        # score the training positives (all fold rows are from this fold)
        model.eval()
        with torch.no_grad():
            pv = model(train_data.x_dict, train_data.edge_index_dict,
                       label_index)
            v_auc = roc_auc_score(label.cpu().numpy(), pv.cpu().numpy())
        if v_auc > best[0]:
            best = (v_auc, ep)
            with torch.no_grad():
                z_full = model.encoder(train_data.x_dict,
                                       train_data.edge_index_dict)
    return z_full, best


def emb_features(z_dict, rna_type, pairs, k=12):
    """Pair features from the GNN embeddings.  Everything is a function of the
    two node embeddings, so the per-fold GNN guard covers them automatically."""
    r, d = pairs[0], pairs[1]
    zr = z_dict[rna_type].cpu().numpy()
    zd = z_dict['drug'].cpu().numpy()
    zr_p, zd_p = zr[r], zd[d]
    dot = (zr_p * zd_p).sum(1)
    norm = np.linalg.norm(zr_p - zd_p, axis=1)
    zr_u = zr_p / (np.linalg.norm(zr_p, axis=1, keepdims=True) + 1e-9)
    zd_u = zd_p / (np.linalg.norm(zd_p, axis=1, keepdims=True) + 1e-9)
    cos = (zr_u * zd_u).sum(1)
    # standardised dot: remove the per-node scale so the tree sees correlation,
    # not magnitude (the magnitude is the drug-popularity signal the explicit
    # degree features already carry).
    zr_s = (zr_p - zr_p.mean(0)) / (zr_p.std(0) + 1e-9)
    zd_s = (zd_p - zd_p.mean(0)) / (zd_p.std(0) + 1e-9)
    dot_a = (zr_s * zd_s).sum(1)
    # leading principal components of the concatenated pair.
    Z = np.concatenate([zr_p, zd_p], axis=1)
    Zc = Z - Z.mean(0)
    _, _, Vh = np.linalg.svd(Zc, full_matrices=False)
    comps = Vh[:k].T                       # (d, k) principal directions
    head = Zc @ comps
    cols = {'emb_dot': dot, 'emb_norm': norm, 'emb_cos': cos,
            'emb_dot_a': dot_a}
    for i in range(k):
        cols[f'emb_pc{i}'] = head[:, i]
    return pd.DataFrame(cols)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--net-dir', required=True)
    p.add_argument('--task', required=True, choices=list(TASKS))
    p.add_argument('--model', default='extratrees',
                   choices=['extratrees', 'gbdt', 'logreg', 'mlp'])
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1])
    p.add_argument('--out', required=True)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = TASKS[args.task]
    key = (t['rna'], args.task, 'drug')
    splits, _, A, S_r, S_d, bridge, attrs = build_inputs(
        args.run_dir, args.net_dir, args.task, 'cpu')
    rows = {n: (d[key].edge_label_index.cpu().numpy(),
                d[key].edge_label.cpu().numpy())
            for n, d in zip(('train', 'val', 'test'), splits)}
    y = {f: rows[f][1] for f in ('val', 'test')}

    # val/test features: base + extra + embeddings from the full-graph GNN.
    z_full, _ = train_embeddings(args.run_dir, args.task, device)
    emb = {f: emb_features(z_full, t['rna'], rows[f][0])
           for f in ('val', 'test')}
    base = {f: featurise(rows[f][0], A, S_r, S_d, bridge,
                         Xr=attrs[0], Xd=attrs[1]) for f in ('val', 'test')}
    X = {f: pd.concat([base[f].reset_index(drop=True),
                   extra_columns(rows[f][0], A, S_r, S_d),
                   emb[f]], axis=1) for f in ('val', 'test')}

    # train features: K-fold, each fold from a per-fold GNN.
    tr_idx, tr_y = rows['train']
    rng = np.random.RandomState(0)
    fold = rng.randint(0, args.folds, size=tr_idx.shape[1])
    parts = []
    for k in range(args.folds):
        m = fold == k
        # per-fold training graph: remove fold-k positives' edges.
        from improve.run import load_splits
        train_data = load_splits(args.run_dir, args.task, ARMS['base'],
                                 device)[0]
        pos_idx = train_data[key].edge_label_index[:, tr_y == 1]
        fold_pos = tr_idx[:, m & (tr_y == 1)]
        ei = train_data[key].edge_index.cpu().numpy()
        rem = set(map(tuple, fold_pos.T.tolist()))
        keep = np.array([tuple(p) not in rem for p in ei.T])
        if not keep.all():
            train_data[key].edge_index = torch.tensor(
                ei[:, keep], device=device)
            # drop the removed edges from the label set too
            keep_l = np.array([tuple(p) not in rem
                               for p in train_data[key].edge_label_index
                               .cpu().numpy().T])
            train_data[key].edge_label_index = train_data[key] \
                .edge_label_index[:, keep_l]
            train_data[key].edge_label = train_data[key].edge_label[keep_l]
        z_fold, _ = train_embeddings_from_data(train_data, args.task, device,
                                               key)
        Ak = A.copy()
        held = tr_idx[:, m & (tr_y == 1)]
        Ak[held[0], held[1]] = 0.0
        f = featurise(tr_idx[:, m], Ak, S_r, S_d, bridge,
                      Xr=attrs[0], Xd=attrs[1])
        f = pd.concat([f.reset_index(drop=True),
                       extra_columns(tr_idx[:, m], Ak, S_r, S_d),
                       emb_features(z_fold, t['rna'], tr_idx[:, m])], axis=1)
        f.index = np.where(m)[0]
        parts.append(f)
    X['train'] = pd.concat(parts).sort_index()

    new_cols = list(emb['val'].columns)
    for label, cols in (('base+extra', list(base['val'].columns)
                         + list(extra_columns(rows['val'][0], A, S_r, S_d).columns)),
                        ('+emb', list(X['train'].columns))):
        acc = {f: {'auc': [], 'aupr': []} for f in ('val', 'test')}
        for seed in args.seeds:
            clf = fitted(args.model, seed)
            clf.fit(X['train'][cols].values, tr_y)
            for f in ('val', 'test'):
                s = clf.predict_proba(X[f][cols].values)[:, 1]
                acc[f]['auc'].append(roc_auc_score(y[f], s))
                acc[f]['aupr'].append(average_precision_score(y[f], s))
        r = dict(task=args.task, model=args.model, set=label,
                 n_feat=len(cols))
        for f in ('val', 'test'):
            for m in ('auc', 'aupr'):
                r[f'{f}_{m}'] = float(np.mean(acc[f][m]))
                r[f'{f}_{m}_std'] = float(np.std(acc[f][m]))
        print(f"  {label:10s} ({len(cols):2d} feat)  "
              f"val {r['val_auc']:.4f}/{r['val_aupr']:.4f}  "
              f"test {r['test_auc']:.4f}/{r['test_aupr']:.4f}")

    pa, pp = PAPER[args.task]['val']
    print(f"\n{args.task} {args.model}: paper S_val {pa:.4f}/{pp:.4f}")
    print('emb features:', ', '.join(new_cols))


if __name__ == '__main__':
    main()