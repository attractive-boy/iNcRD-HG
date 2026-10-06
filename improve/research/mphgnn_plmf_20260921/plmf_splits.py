# -*- coding: utf-8 -*-
"""Index-level copies of the two PLMF-MDA benchmark protocols used throughout
improve/research/plmf_baselines_20260920/plmf_data.py, so the MPHGNN runs share
bit-identical splits with PLMF-MDA and its four reproduced baselines.
  5cv    : positives + equal sampled negatives (RandomState(seed)), shuffled,
           np.array_split into 5 folds; fold f is the test fold.   (paper)
  single : released-code 80/20 split (np.random.seed(seed) -> shuffle pos ->
           shuffle neg -> last 20% test).                          (official code)
Returns (train_idx[N,2], y_train, test_idx[M,2], y_test) with (miRNA, drug) ids."""
import os
import numpy as np
import pandas as pd

PLMF = '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/new5_repro_20260919/repos/PLMF-MDA-main/data/unpacked2/data'
RECORDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plmf_xrel_20260922', 'data', 'full_{}', 'data', 'pair_records.pkl')
SUF = {'MDR': 'R', 'MDS': 'S'}
N_FOLDS = 5
TEST_RATIO = 0.2


def load_dataset(dataset):
    suf = SUF[dataset]
    d = os.path.join(PLMF, dataset)
    mir = pd.read_excel(os.path.join(d, f'miRNA_ID_{suf}.xlsx'))
    dru = pd.read_excel(os.path.join(d, f'drug_ID_{suf}.xlsx'))
    pos = np.loadtxt(os.path.join(d, f'pos_MDA_{suf}.edgelist'), dtype=np.int64)
    neg = np.loadtxt(os.path.join(d, f'neg_MDA_{suf}.edgelist'), dtype=np.int64)
    return mir, dru, pos, neg


def _balanced_pairs(pos, neg, seed):
    rng = np.random.RandomState(seed)
    n = len(pos)
    neg_sel = neg[rng.permutation(len(neg))[:n]]
    idx = np.concatenate([pos, neg_sel], axis=0)
    y = np.concatenate([np.ones(n, dtype=np.float64), np.zeros(n, dtype=np.float64)])
    order = rng.permutation(len(idx))
    return idx[order], y[order]


def fold_5cv(dataset, fold, seed=42, n_folds=N_FOLDS):
    _, _, pos, neg = load_dataset(dataset)
    idx, y = _balanced_pairs(pos, neg, seed)
    chunks = np.array_split(np.arange(len(idx)), n_folds)
    te = chunks[fold]
    tr = np.concatenate([chunks[f] for f in range(n_folds) if f != fold])
    return idx[tr], y[tr], idx[te], y[te]


def cold_drug_split(dataset, fold, seed=42, n_folds=N_FOLDS):
    """Cold-drug (inductive) 5-fold split == ../plmf_frontier_20260922/cold_probe.py --protocol ldo:
    drugs are binned by positive degree and split into 5 folds (StratifiedKFold); test = positives of
    the test drugs + an equal number of sampled (miRNA, test-drug) zeros; train = positives of the
    remaining drugs + equal sampled zeros.  Test drugs have zero training positives."""
    from sklearn.model_selection import StratifiedKFold
    mir, dru, pos, neg = load_dataset(dataset)
    nr, nd = len(mir), len(dru)
    A = np.zeros((nr, nd), np.int8); A[pos[:, 0], pos[:, 1]] = 1
    bins = np.digitize(A.sum(0), [2, 6, 21, 101])
    test_drugs = list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(nd), bins))[fold][1]
    is_te = np.zeros(nd, bool); is_te[test_drugs] = True
    rng = np.random.RandomState(seed * 10 + fold)

    def make(mask):
        P = np.argwhere((A == 1) & mask[None, :]); Z = np.argwhere((A == 0) & mask[None, :])
        N = Z[rng.permutation(len(Z))[:len(P)]]
        idx = np.concatenate([P, N]); y = np.concatenate([np.ones(len(P)), np.zeros(len(N))])
        o = rng.permutation(len(idx)); return idx[o], y[o]

    tr, ytr = make(~is_te); te, yte = make(is_te)
    return tr, ytr, te, yte


def study_split_primary(dataset, fold, seed=42, n_folds=N_FOLDS):
    """Primary-study grouped 5-fold split (== ../plmf_xrel_20260922/study_split2.py --group primary):
    each positive is assigned to the fold of its most specific PMID (fewest reported pairs); negatives
    are random.  Same-study pairs therefore never straddle the train/test boundary."""
    import pickle
    mir, dru, pos, neg = load_dataset(dataset)
    names = mir['miRNA_Name'].astype(str).str.upper().values
    dbs = dru['DrugBank_ID'].astype(str).values
    recs = pickle.load(open(RECORDS.format(dataset), 'rb'))
    pm = {k: recs[(names[i], dbs[j])]['pmids'] for k, (i, j) in enumerate(map(tuple, pos)) if (names[i], dbs[j]) in recs}
    size = {}
    for v in pm.values():
        for p in v:
            size[p] = size.get(p, 0) + 1
    grp = np.array([min(pm[k], key=lambda p: (size[p], p)) if k in pm else -(k + 1) for k in range(len(pos))])
    rng = np.random.RandomState(seed); sizes = pd.Series(grp).value_counts()
    order = sizes.index.values.copy(); rng.shuffle(order); order = sorted(order, key=lambda c: -sizes[c])
    fold_of, load = {}, np.zeros(n_folds)
    for c in order:
        f = int(np.argmin(load + rng.rand(n_folds) * 0.5)); fold_of[c] = f; load[f] += sizes[c]
    pos_fold = np.array([fold_of[c] for c in grp])
    idx, y = _balanced_pairs(pos, neg, seed); neg_sel = idx[y == 0]; neg_fold = rng.randint(0, n_folds, len(neg_sel))
    tr = np.concatenate([pos[pos_fold != fold], neg_sel[neg_fold != fold]])
    ytr = np.concatenate([np.ones((pos_fold != fold).sum()), np.zeros((neg_fold != fold).sum())])
    te = np.concatenate([pos[pos_fold == fold], neg_sel[neg_fold == fold]])
    yte = np.concatenate([np.ones((pos_fold == fold).sum()), np.zeros((neg_fold == fold).sum())])
    return tr, ytr, te, yte


def single_split(dataset, seed=0, test_ratio=TEST_RATIO):
    _, _, pos, neg = load_dataset(dataset)
    pos = pos.copy(); neg = neg.copy()
    np.random.seed(seed)
    np.random.shuffle(pos)
    np.random.shuffle(neg)
    neg_sel = neg[:len(pos)]
    te_size = int(test_ratio * len(pos))
    tr_idx = np.concatenate([pos[:-te_size], neg_sel[:-te_size]], axis=0)
    te_idx = np.concatenate([pos[-te_size:], neg_sel[-te_size:]], axis=0)
    y_tr = np.concatenate([np.ones(len(pos) - te_size), np.zeros(len(pos) - te_size)])
    y_te = np.concatenate([np.ones(te_size), np.zeros(te_size)])
    return tr_idx, y_tr, te_idx, y_te


def get_split(dataset, split, fold=0, seed=42):
    if split == '5cv':
        return fold_5cv(dataset, fold, seed)
    if split == 'study':
        return study_split_primary(dataset, fold, seed)
    if split == 'cold':
        return cold_drug_split(dataset, fold, seed)
    return single_split(dataset, seed)


if __name__ == '__main__':
    # equality check against plmf_data.py (pair-level, since it returns (smiles, seq))
    import sys
    sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/plmf_baselines_20260920')
    import plmf_data as P
    for ds in ('MDR', 'MDS'):
        mir, dru, pos, neg = load_dataset(ds)
        seqs = mir['miRNA_Sequence'].astype(str).tolist(); smiles = dru['SMILES'].astype(str).tolist()
        for split, seeds in (('5cv', [42]), ('single', [0, 1, 2, 3, 4])):
            for sd in seeds:
                for f in (range(5) if split == '5cv' else [0]):
                    a = P.load_fold_5cv(ds, f, sd) if split == '5cv' else P.load_single_split(ds, sd)
                    tr, ytr, te, yte = get_split(ds, split, f, sd)
                    assert [(smiles[j], seqs[i]) for i, j in tr] == a[0] and np.array_equal(ytr, a[1])
                    assert [(smiles[j], seqs[i]) for i, j in te] == a[2] and np.array_equal(yte, a[3])
        print(ds, 'splits identical to plmf_data.py')
