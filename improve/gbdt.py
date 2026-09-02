# A model class that is not a GNN: heterogeneous link-prediction features + GBDT.
#
# Every ensemble member in this repo has been a variant of the same two-layer
# GeneralConv encoder.  Widths, dropouts, aggregations and seeds change the
# weights but not the inductive bias, so the members fail on the same edges and
# uniform averaging buys +0.0005.  A gradient-boosted tree over explicit
# link-prediction features has a different bias entirely: it sees
# similarity-weighted collaborative scores and path counts directly, as numbers,
# instead of having to recover them from message passing.
#
# LEAKAGE is the whole difficulty here.  A training positive (r,d) is an edge of
# the very association matrix its features are computed from, so a
# collaborative feature would read its own label.  Two guards:
#
#   * the similarity matrices carry a zero diagonal, so S@A at (r,d) never picks
#     up A[r,d] through the r=r' term;
#   * training rows get K-fold feature generation -- for fold k, every feature
#     is computed from the association matrix with fold k's positives removed.
#     Validation and test rows use the full training matrix, which is what they
#     would see at deployment.
#
# The control for all of this is a label permutation, and it has to be run MANY
# times, not once.  The null distribution of validation AUC under permutation
# has a standard deviation of 0.09-0.13 here -- 3801 training rows and a 1263-row
# validation fold are small enough that a noise-fitted model drifts a long way
# from 0.5 by chance.  Over 20 permutations the null comes out at 0.5025 +- 0.128
# (extratrees), 0.4554 +- 0.090 (gbdt), 0.5153 +- 0.123 (logreg), and the real
# scores sit above every draw.  A single permutation returning 0.65 says nothing;
# it is 1.1 standard deviations from the mean of the null.
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.run import TASKS

OTHER = {'LncDrug': ('lncRNA', 'miRNA', 'MiDrug'),
         'MiDrug': ('miRNA', 'lncRNA', 'LncDrug')}
EPS = 1e-9


def dense(edge_index, n_src, n_dst):
    a = np.zeros((n_src, n_dst), dtype=np.float32)
    ei = edge_index.cpu().numpy()
    a[ei[0], ei[1]] = 1.0
    return a


def sim_matrix(path, n):
    m = np.abs(pd.read_csv(path, header=None).values).astype(np.float32)
    m = np.nan_to_num(m, nan=0.0)
    m = m[:n, :n]
    np.fill_diagonal(m, 0.0)          # guard 1: no self term in S @ A
    return m


def featurise(pairs, A, S_r, S_d, bridge, n_svd=16, Xr=None, Xd=None, topk=5):
    """Features for an array of (rna, drug) index pairs, from association A.

    Everything is a function of `A`, so the K-fold guard in main() covers every
    feature automatically -- including the latent factors, which is why the SVD
    is computed here and not once outside.

    `Xr`/`Xd` are the raw node attributes (expression profiles, ChemBERTa).
    They add the block `improve/feature_groups.py` showed MiDrug responds to:
    that task gains +0.0019 from continuous similarity against LncDrug's
    +0.0003, and the attribute block is the same kind of information one step
    less processed.  The GNN holds these vectors but its decoder never compares
    a pair of them; the tree can compare but has only ever seen the released
    Kendall matrix derived from them.
    """
    r, d = pairs[0], pairs[1]
    deg_r, deg_d = A.sum(1), A.sum(0)

    SrA = S_r @ A                      # similarity-weighted collaborative, ncRNA
    ASd = A @ S_d                      # ... drug side
    SrASd = SrA @ S_d
    AtA = A.T @ A
    AAtA = A @ AtA                     # 3-hop bipartite path count
    A5 = AAtA @ AtA                    # 5-hop, the Katz truncation
    srow, dcol = S_r.sum(1), S_d.sum(1)

    # Latent factors of the association matrix: the matrix-factorisation view
    # of the same data, which trees cannot construct from the raw counts.
    k = min(n_svd, min(A.shape) - 1)
    U, sig, Vt = np.linalg.svd(A, full_matrices=False)
    U, V = U[:, :k] * sig[:k], Vt[:k].T
    svd_dot = (U[r] * V[d]).sum(1)

    # Degree of the neighbourhood, not of the endpoint: a drug attached to
    # well-connected ncRNAs is different from one attached to isolated ones.
    nbr_deg_r = (A @ deg_d) / (deg_r + EPS)
    nbr_deg_d = (A.T @ deg_r) / (deg_d + EPS)

    # max similarity to an ncRNA that IS associated with d, and vice versa.
    # Masked maxima, not means: one very similar neighbour is the signal.
    maxsim_r = np.stack([(S_r * A[:, j][None, :]).max(1)
                         for j in range(A.shape[1])], axis=1)
    maxsim_d = (A[:, :, None] * S_d[None, :, :]).max(1)

    cols = {
        'log_deg_r': np.log1p(deg_r[r]),
        'log_deg_d': np.log1p(deg_d[d]),
        'pref_attach': np.log1p(deg_r[r]) * np.log1p(deg_d[d]),
        'collab_r': SrA[r, d],
        'collab_r_norm': SrA[r, d] / (srow[r] + EPS),
        'collab_d': ASd[r, d],
        'collab_d_norm': ASd[r, d] / (dcol[d] + EPS),
        'collab_both': SrASd[r, d],
        'path3': AAtA[r, d],
        'path3_norm': AAtA[r, d] / (deg_r[r] * deg_d[d] + EPS),
        'path5': A5[r, d],
        'path5_norm': A5[r, d] / (deg_r[r] * deg_d[d] + EPS),
        'svd_dot': svd_dot,
        'nbr_deg_r': nbr_deg_r[r],
        'nbr_deg_d': nbr_deg_d[d],
        'maxsim_r': maxsim_r[r, d],
        'maxsim_d': maxsim_d[r, d],
        'bridge': bridge[r, d],
        'bridge_norm': bridge[r, d] / (deg_d[d] + EPS),
    }

    # Similarity block, second order and top-k.  `maxsim_*` takes the single
    # most similar associated neighbour; a mean over the top-k is less brittle,
    # and S @ S @ A reaches neighbours-of-neighbours, which no single-hop
    # similarity feature sees.
    def topk_mean(S, M):
        V = S[:, :, None] * M[None, :, :]
        k = min(topk, V.shape[1])
        return np.sort(V, axis=1)[:, -k:, :].mean(1)

    cols['topk_sim_r'] = topk_mean(S_r, A)[r, d]
    cols['topk_sim_d'] = topk_mean(S_d, A.T)[d, r]
    cols['collab_r2'] = (S_r @ (S_r @ A))[r, d]
    cols['collab_d2'] = ((A @ S_d) @ S_d)[r, d]
    cols['simdeg_r'] = (S_r @ deg_r)[r]
    cols['simdeg_d'] = (S_d @ deg_d)[d]

    if Xr is not None and Xd is not None:
        # Raw-attribute pair features: how close is this drug to the drugs the
        # ncRNA already hits, and vice versa.  Cosine in each attribute space.
        def unit(X):
            return X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)
        Ur, Ud = unit(Xr), unit(Xd)
        prof_d = (A @ Ud) / (deg_r[:, None] + EPS)      # ncRNA -> mean drug vec
        prof_r = (A.T @ Ur) / (deg_d[:, None] + EPS)     # drug  -> mean ncRNA vec
        cols['attr_cos_d'] = (unit(prof_d)[r] * Ud[d]).sum(1)
        cols['attr_cos_r'] = (unit(prof_r)[d] * Ur[r]).sum(1)
        cols['attr_maxcos_d'] = np.array(
            [(Ud[d[i]] @ Ud[A[r[i]] > 0].T).max() if (A[r[i]] > 0).any() else 0.0
             for i in range(len(r))])
        cols['attr_maxcos_r'] = np.array(
            [(Ur[r[i]] @ Ur[A[:, d[i]] > 0].T).max()
             if (A[:, d[i]] > 0).any() else 0.0 for i in range(len(r))])
    return pd.DataFrame(cols)


def build_inputs(run_dir, net_dir, task, device):
    t = TASKS[task]
    rna, other_rna, other_task = OTHER[task]
    key = (rna, task, 'drug')
    splits = [torch.load(f'{run_dir}/3.heter_data/{task}_{s}_data.pth',
                         map_location='cpu') for s in ('train', 'val', 'test')]
    train = splits[0]
    n_r, n_d = train[rna].num_nodes, train['drug'].num_nodes
    A = dense(train[key].edge_index, n_r, n_d)

    sim_file = {'lncRNA': 'LncSimMat_ken.csv', 'miRNA': 'MiSimMat_ken.csv'}[rna]
    S_r = sim_matrix(f'{net_dir}/{sim_file}', n_r)
    S_d = sim_matrix(f'{net_dir}/drugSimMat.csv', n_d)

    # The pan-ncRNA path: ncRNA -> other ncRNA -> drug, as an explicit count.
    # Both LncMi and the other relation's drug edges sit in this task's graph
    # untouched, so no held-out edge of THIS task can enter it.
    n_o = train[other_rna].num_nodes
    lm = train[('lncRNA', 'LncMi', 'miRNA')].edge_index
    L2M = dense(lm, train['lncRNA'].num_nodes, train['miRNA'].num_nodes)
    O2D = dense(train[(other_rna, other_task, 'drug')].edge_index, n_o, n_d)
    bridge = (L2M @ O2D) if task == 'LncDrug' else (L2M.T @ O2D)
    # Raw node attributes, for the attribute-pair block.  These are exactly what
    # the GNN receives as x_dict.
    attrs = (train[rna].x.cpu().numpy().astype(np.float32),
             train['drug'].x.cpu().numpy().astype(np.float32))
    return splits, key, A, S_r, S_d, bridge, attrs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--net-dir', required=True)
    p.add_argument('--task', required=True, choices=list(TASKS))
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1, 2, 3])
    p.add_argument('--leaky', action='store_true',
                   help='skip K-fold feature generation, to show what the '
                        'leak is worth (a control, not a result)')
    p.add_argument('--permute-labels', type=int, default=None,
                   help='seed for permuting the training labels. Run this over '
                        'many seeds to build the null distribution; a single '
                        'draw is uninformative (null sd is ~0.1 here).')
    p.add_argument('--dump-scores', default=None)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    splits, key, A, S_r, S_d, bridge, attrs = build_inputs(
        args.run_dir, args.net_dir, args.task, 'cpu')
    train, val, test = splits
    rows = {}
    for name, d in (('train', train), ('val', val), ('test', test)):
        rows[name] = (d[key].edge_label_index.cpu().numpy(),
                      d[key].edge_label.cpu().numpy())

    # Validation and test see the full training association matrix.
    X = {n: featurise(rows[n][0], A, S_r, S_d, bridge, Xr=attrs[0],
                      Xd=attrs[1]) for n in ('val', 'test')}

    # Training rows are featurised K-fold, each fold from a matrix with that
    # fold's positives removed.
    tr_idx, tr_y = rows['train']
    if args.leaky:
        X['train'] = featurise(tr_idx, A, S_r, S_d, bridge, Xr=attrs[0], Xd=attrs[1])
    else:
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

    res, scores = [], {}
    y_fit = tr_y
    if args.permute_labels is not None:
        y_fit = np.random.RandomState(args.permute_labels).permutation(tr_y)
    for seed in args.seeds:
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=False, random_state=seed)
        clf.fit(X['train'].values, y_fit)
        r = dict(task=args.task, seed=seed, leaky=args.leaky,
                 permuted=args.permute_labels)
        for f in ('val', 'test'):
            s = clf.predict_proba(X[f].values)[:, 1]
            scores.setdefault(f, []).append(s)
            r[f'{f}_auc'] = roc_auc_score(rows[f][1], s)
            r[f'{f}_aupr'] = average_precision_score(rows[f][1], s)
        res.append(r)
        print(f"  seed={seed:<4d} val {r['val_auc']:.4f}/{r['val_aupr']:.4f} "
              f"test {r['test_auc']:.4f}/{r['test_aupr']:.4f}", flush=True)

    df = pd.DataFrame(res)
    df.to_csv(args.out, index=False)
    print(f"\n{args.task} GBDT ({args.folds}-fold features"
          f"{', LEAKY' if args.leaky else ''}), {len(args.seeds)} seeds")
    for f in ('val', 'test'):
        print(f"  {f}: AUC {df[f'{f}_auc'].mean():.4f} "
              f"AUPR {df[f'{f}_aupr'].mean():.4f}")
    if args.dump_scores:
        np.savez(args.dump_scores,
                 **{f: np.mean(scores[f], axis=0) for f in ('val', 'test')},
                 y_val=rows['val'][1], y_test=rows['test'][1])
        print(f'wrote {args.dump_scores}')


if __name__ == '__main__':
    main()
