# The drug-target bipartite channel: a genuinely orthogonal information entry.
#
# Every existing drug-side channel is a function of molecular structure:
# `drugSimMat.csv` is Tanimoto over Morgan fingerprints, the ChemBERTa cosine
# view in `simviews.py` is the same molecule read by a different encoder, and
# the ATC view is a therapeutic *classification*.  None of them sees the
# *protein* that the drug acts on, and drug response is mediated by targets,
# not by fingerprints.
#
# `improve/drug_target.py` already fetched the ChEMBL mechanism data
# (`14.drug/chembl.json`): 103 of the 216 drugs carry >=1 mechanism target,
# 116 distinct targets, 200 edges.  It was used only to compute two pairwise
# similarity matrices (S_atc, S_target); the target-overlap Jaccard came in at
# 0.41% density and was dropped.  But that verdict judged the target data as a
# *replacement* for the fingerprint matrix.  This file adds it as an *extra*
# relation instead -- a bipartite `drug -> target` edge, collapsed to a
# target-overlap drug matrix `T` -- and measures the additive signal on top of
# the released 29 features.
#
# Orthogonality is the point.  On the 360 drug pairs that share a target,
# Spearman(target-overlap, Tanimoto) = 0.075 and Pearson = 0.073 over all
# nonempty pairs: the two matrices order drugs almost independently.  If target
# overlap predicted associations where fingerprints do not, the features below
# carry information the model has never seen.
#
# Leakage: T is a node attribute -- a function of drug identity, computed from
# ChEMBL annotations that read no association edge.  It sits outside the K-fold
# guard exactly like the released S_d, and every new feature is a function of
# (A, S_r, S_d, T), so the same `gbdt.py` K-fold protocol covers it.
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.altmodels import fitted
from improve.gbdt import EPS, build_inputs, featurise
from improve.run import TASKS

PAPER = {'LncDrug': dict(val=(0.9834, 0.9545), test=(0.9857, 0.9585)),
         'MiDrug': dict(val=(0.9555, 0.9183), test=(0.9564, 0.9165))}


def build_target_matrix(net_dir, chembl_json, n_drug):
    """drug x drug target-overlap matrix T from the ChEMBL mechanism cache.

    Returns (T, meta) where meta reports coverage.  T[i, j] is the number of
    distinct protein targets drugs i and j share (Jaccard would be a symmetric
    renormalisation; the raw count is what the tree features want).  Diagonal
    zeroed for the S @ A guard, mirroring `gbdt.sim_matrix`.
    """
    d = json.load(open(chembl_json))
    mol, mech = d.get('molecule', {}), d.get('mechanism', {})
    drugs = pd.read_csv(os.path.join(net_dir, 'Drug.csv'))
    name2chembl = {name: v['chembl_id'] for name, v in mol.items()}

    B = np.zeros((n_drug, 0), dtype=np.float32)
    target_sets = []
    for name in drugs['Drug_Name']:
        cid = name2chembl.get(name)
        ts = set()
        for m in mech.get(cid, []):
            if m.get('target'):
                ts.add(m['target'])
        target_sets.append(ts)

    all_t = sorted(set().union(*target_sets))
    t2idx = {t: k for k, t in enumerate(all_t)}
    B = np.zeros((n_drug, len(all_t)), dtype=np.float32)
    for i, ts in enumerate(target_sets):
        for t in ts:
            B[i, t2idx[t]] = 1.0

    T = (B @ B.T).astype(np.float32)
    np.fill_diagonal(T, 0.0)

    meta = dict(
        n_covered=int((B.sum(1) > 0).sum()),
        n_targets=B.shape[1],
        n_edges=int(B.sum()),
        n_overlap_pairs=int((T > 0).sum()),
        overlap_density=float((T > 0).sum() / (n_drug * n_drug - n_drug)),
    )
    return T, meta


def target_columns(pairs, A, S_r, S_d, T, topk=5):
    """Target-overlap features, all functions of (A, S_r, S_d, T).

    The families mirror `gbdt.featurise`: the same collaborative/path/max
    statistics, but with the target-overlap matrix T in the place of the
    fingerprint matrix S_d.  Where the fingerprint and the target disagree,
    these columns split the pairs that the base 29 ranks the same.
    """
    r, d = pairs[0], pairs[1]
    deg_r, deg_d = A.sum(1), A.sum(0)
    srow = S_r.sum(1)
    dcol = T.sum(1)

    AT = A @ T                          # ncRNA's drugs -> target-shared drugs
    SrAT = S_r @ AT

    # top-k mean target overlap to d's associated drugs
    V = A[:, :, None] * T[None, :, :]   # [r, nbr_drug, candidate]
    k = min(topk, V.shape[1])
    topk_d = np.sort(V, axis=1)[:, -k:, :].mean(1)

    return pd.DataFrame({
        'tgt_collab_d': AT[r, d],
        'tgt_collab_d_norm': AT[r, d] / (dcol[d] + EPS),
        'tgt_collab_both': SrAT[r, d],
        'tgt_topk_d': topk_d[r, d],
        'tgt_maxsim_d': (A[:, :, None] * T[None, :, :]).max(1)[r, d],
        'tgt_simdeg_d': (T @ deg_d)[d],
        'tgt_simdeg_r': (S_r @ deg_r)[r],
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--net-dir', required=True)
    p.add_argument('--chembl-json', required=True,
                   help='14.drug/chembl.json from improve/drug_target.py')
    p.add_argument('--task', required=True, choices=list(TASKS))
    p.add_argument('--model', default='extratrees',
                   choices=['extratrees', 'gbdt', 'logreg', 'mlp'])
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1, 2, 3])
    p.add_argument('--out', required=True)
    args = p.parse_args()

    splits, key, A, S_r, S_d, bridge, attrs = build_inputs(
        args.run_dir, args.net_dir, args.task, 'cpu')
    n_drug = S_d.shape[0]
    T, meta = build_target_matrix(args.net_dir, args.chembl_json, n_drug)
    print(f"target channel: {meta}")

    rows = {n: (d[key].edge_label_index.cpu().numpy(),
                d[key].edge_label.cpu().numpy())
            for n, d in zip(('train', 'val', 'test'), splits)}
    y = {f: rows[f][1] for f in ('val', 'test')}

    base = {f: featurise(rows[f][0], A, S_r, S_d, bridge,
                         Xr=attrs[0], Xd=attrs[1]) for f in ('val', 'test')}
    extra = {f: target_columns(rows[f][0], A, S_r, S_d, T) for f in ('val', 'test')}
    X = {f: pd.concat([base[f], extra[f]], axis=1) for f in ('val', 'test')}

    tr_idx, tr_y = rows['train']
    rng = np.random.RandomState(0)
    fold = rng.randint(0, args.folds, size=tr_idx.shape[1])
    parts = []
    for k in range(args.folds):
        m = fold == k
        Ak = A.copy()
        held = tr_idx[:, m & (tr_y == 1)]
        Ak[held[0], held[1]] = 0.0
        f = pd.concat([featurise(tr_idx[:, m], Ak, S_r, S_d, bridge,
                                 Xr=attrs[0], Xd=attrs[1]),
                       target_columns(tr_idx[:, m], Ak, S_r, S_d, T)], axis=1)
        f.index = np.where(m)[0]
        parts.append(f)
    X['train'] = pd.concat(parts).sort_index()

    new_cols = list(extra['val'].columns)
    for label, cols in (('base29', list(base['val'].columns)),
                        ('+target', list(X['train'].columns))):
        acc = {f: {'auc': [], 'aupr': []} for f in ('val', 'test')}
        for seed in args.seeds:
            clf = fitted(args.model, seed)
            clf.fit(X['train'][cols].values, tr_y)
            for f in ('val', 'test'):
                s = clf.predict_proba(X[f][cols].values)[:, 1]
                acc[f]['auc'].append(roc_auc_score(y[f], s))
                acc[f]['aupr'].append(average_precision_score(y[f], s))
        r = dict(task=args.task, model=args.model, set=label, n_feat=len(cols))
        for f in ('val', 'test'):
            for m in ('auc', 'aupr'):
                r[f'{f}_{m}'] = float(np.mean(acc[f][m]))
                r[f'{f}_{m}_std'] = float(np.std(acc[f][m]))
        print(f"  {label:10s} ({len(cols):2d} feat)  "
              f"val {r['val_auc']:.4f}/{r['val_aupr']:.4f}  "
              f"test {r['test_auc']:.4f}/{r['test_aupr']:.4f}")

    pa, pp = PAPER[args.task]['val']
    print(f"\n{args.task} {args.model}: paper S_val {pa:.4f}/{pp:.4f}")
    print('new features:', ', '.join(new_cols))

    res = []
    for label, cols in (('base29', list(base['val'].columns)),
                        ('+target', list(X['train'].columns))):
        acc = {f: {'auc': [], 'aupr': []} for f in ('val', 'test')}
        for seed in args.seeds:
            clf = fitted(args.model, seed)
            clf.fit(X['train'][cols].values, tr_y)
            for f in ('val', 'test'):
                s = clf.predict_proba(X[f][cols].values)[:, 1]
                acc[f]['auc'].append(roc_auc_score(y[f], s))
                acc[f]['aupr'].append(average_precision_score(y[f], s))
        row = dict(task=args.task, model=args.model, set=label, n_feat=len(cols))
        for f in ('val', 'test'):
            for m in ('auc', 'aupr'):
                row[f'{f}_{m}'] = float(np.mean(acc[f][m]))
                row[f'{f}_{m}_std'] = float(np.std(acc[f][m]))
        res.append(row)
    pd.DataFrame(res).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()