# -*- coding: utf-8 -*-
"""Data + split utilities for reproducing PLMF-MDA's baseline comparison (Fig 3).

Benchmark (PLMF-MDA, PLoS Comput Biol 2026, github.com/sheng-n/PLMF-MDA):
  MDRdataset: 1317 miRNA x 105 drug, 5411 resistance associations
  MDSdataset: 1252 miRNA x 140 drug, 5054 sensitivity associations
  Protocol in the paper: positives + equal sampled negatives, 5-fold CV.

The released code (`code/data_preprocess.py::load_data`) implements a single
80/20 split instead of 5-cv; both are provided here:
  * load_fold_5cv  -> paper's stated 5-fold protocol (primary)
  * load_single_split -> released-code protocol (directly comparable to the
                          already-reproduced PLMF-MDA reference, seeds 0..4)

Pairs are returned as (smiles, mirna_sequence) so the four baseline models from
`improve/research/dlstmda_baselines_20260919/models.py` can consume them unchanged.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DLST = os.path.join(os.path.dirname(HERE), 'dlstmda_baselines_20260919')
PLMF_DATA = os.path.join(
    os.path.dirname(HERE), 'new5_repro_20260919', 'repos', 'PLMF-MDA-main',
    'data', 'unpacked2', 'data')

# reuse the DLST baseline featurisers / metrics (models.py imports `common`)
sys.path.insert(0, DLST)
from common import seed_it, seq2int, smiles2int, smile_to_graph, compute_metrics  # noqa: E402

DATASETS = {'MDR': 'R', 'MDS': 'S'}
N_FOLDS = 5
TEST_RATIO = 0.2


def load_dataset(dataset):
    """Return (mirna_seqs, drug_smiles, pos, neg_pool) for MDR/MDS."""
    suf = DATASETS[dataset]
    d = os.path.join(PLMF_DATA, dataset)
    mir = pd.read_excel(os.path.join(d, f'miRNA_ID_{suf}.xlsx'))
    dru = pd.read_excel(os.path.join(d, f'drug_ID_{suf}.xlsx'))
    pos = np.loadtxt(os.path.join(d, f'pos_MDA_{suf}.edgelist'), dtype=np.int64)
    neg = np.loadtxt(os.path.join(d, f'neg_MDA_{suf}.edgelist'), dtype=np.int64)
    seqs = mir['miRNA_Sequence'].astype(str).tolist()
    smiles = dru['SMILES'].astype(str).tolist()
    return seqs, smiles, pos, neg


def _to_pairs(idx, seqs, smiles):
    return [(smiles[j], seqs[i]) for i, j in idx]


def _balanced_pairs(pos, neg, seed):
    """positives + equal-count sampled negatives, shuffled. Returns idx, labels."""
    rng = np.random.RandomState(seed)
    n = len(pos)
    neg_sel = neg[rng.permutation(len(neg))[:n]]
    idx = np.concatenate([pos, neg_sel], axis=0)
    y = np.concatenate([np.ones(n, dtype=np.float64), np.zeros(n, dtype=np.float64)])
    order = rng.permutation(len(idx))
    return idx[order], y[order]


def load_fold_5cv(dataset, fold, seed=42, n_folds=N_FOLDS):
    """Paper protocol: balanced pos/neg split into `n_folds`; fold `fold` is test."""
    seqs, smiles, pos, neg = load_dataset(dataset)
    idx, y = _balanced_pairs(pos, neg, seed)
    chunks = np.array_split(np.arange(len(idx)), n_folds)
    te = chunks[fold]
    tr = np.concatenate([chunks[f] for f in range(n_folds) if f != fold])
    return (_to_pairs(idx[tr], seqs, smiles), y[tr],
            _to_pairs(idx[te], seqs, smiles), y[te])


def load_single_split(dataset, seed=0, test_ratio=TEST_RATIO):
    """Released-code protocol (`load_data`): shuffle positives, negatives = n_pos,
    80/20 split of the balanced pool.  Mirrors the official global-RNG order
    (`np.random.seed(seed)` -> shuffle positives -> shuffle negative pool) so the
    split is bit-identical to the PLMF-MDA reproduction in new5_repro_20260919."""
    seqs, smiles, pos, neg = load_dataset(dataset)
    pos = pos.copy()
    neg = neg.copy()
    np.random.seed(seed)
    np.random.shuffle(pos)
    np.random.shuffle(neg)
    neg_sel = neg[:len(pos)]
    te_size = int(test_ratio * len(pos))
    tr_idx = np.concatenate([pos[:-te_size], neg_sel[:-te_size]], axis=0)
    te_idx = np.concatenate([pos[-te_size:], neg_sel[-te_size:]], axis=0)
    y_tr = np.concatenate([np.ones(len(pos) - te_size), np.zeros(len(pos) - te_size)])
    y_te = np.concatenate([np.ones(te_size), np.zeros(te_size)])
    return (_to_pairs(tr_idx, seqs, smiles), y_tr,
            _to_pairs(te_idx, seqs, smiles), y_te)


def all_pairs(dataset, split, seed=42):
    """Every unique (smiles, seq) pair across all folds -- for GCNNMMA vocab."""
    seqs, smiles, pos, neg = load_dataset(dataset)
    if split == '5cv':
        pairs = []
        for f in range(N_FOLDS):
            tr, _, te, _ = load_fold_5cv(dataset, f, seed)
            pairs += tr + te
    else:
        tr, _, te, _ = load_single_split(dataset, seed)
        pairs = tr + te
    return list(dict.fromkeys(pairs))
