# -*- coding: utf-8 -*-
"""Run iNcRD-HG's strongest portable reference (frozen TabICLv2 graph-context + anchor
shrinkage) on MSFFMDI's ncDR / RNAInter benchmark, on the same clean 5-fold CV used for
the reproduced baselines.

Reuses the graph-context pair encoder from
`../dlstmda_strongest_20260920/features.py` and the fold construction from
`../msffmdi_baselines_20260920/` (deduped positives + equal negatives, seed 42).
"""
import argparse
import csv
import os
import pickle
import statistics as st
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
STRONG = os.path.join(BASE, 'dlstmda_strongest_20260920')
MSFFM = os.path.join(BASE, 'msffmdi_baselines_20260920')
sys.path.insert(0, STRONG)
sys.path.insert(0, MSFFM)
sys.path.insert(0, HERE)

import features as F  # noqa: E402
import msffmdi_data as MD  # noqa: E402
import msffmdi_models as MM  # noqa: E402
import run_msffmdi as RM  # noqa: E402

RESULTS = os.path.join(HERE, 'results')
LOGS = os.path.join(HERE, 'logs')


def build_data(dataset, seed=42):
    data = MM.build_splits(dataset, seed)
    names = data['names']            # miRNA names (keys)
    cids = data['cids']              # drug cids (keys)
    seqs, smis = data['seqs'], data['smis']
    mirnas = list(dict.fromkeys(seqs[q] for q in names))    # unique miRNA sequences
    drugs = list(dict.fromkeys(smis[c] for c in cids))      # unique drug SMILES
    folds = {}
    for f, (tr, te) in enumerate(data['folds']):
        def pack(idx):
            pairs = [(smis[cids[j]], seqs[names[i]]) for i, j in data['pairs'][idx]]
            return pairs, data['y'][idx]
        folds[(f, 'train')] = pack(tr)
        folds[(f, 'test')] = pack(te)
    return mirnas, drugs, folds


def cache_path(dataset):
    return os.path.join(HERE, f'graph_cache_{dataset}.npz')


def build_cache(dataset):
    names, cids, _ = build_data(dataset)
    xr = np.stack([F.kmer_features(q) for q in names]).astype(np.float32)
    xd = np.stack([F.morgan(s) for s in cids]).astype(np.float32)
    np.savez_compressed(cache_path(dataset), xr=xr, xd=xd,
                        mirnas=np.array(names, dtype=object), drugs=np.array(cids, dtype=object))
    return cache_path(dataset)


def load_cache(dataset):
    d = np.load(cache_path(dataset), allow_pickle=True)
    return d['xr'].astype(np.float32), d['xd'].astype(np.float32), list(d['mirnas']), list(d['drugs'])


def folds_pkl(dataset):
    return os.path.join(HERE, f'folds_{dataset}.pkl')


def build_folds(dataset):
    names, cids, folds = build_data(dataset)
    with open(folds_pkl(dataset), 'wb') as fh:
        pickle.dump((names, cids, folds), fh)
    return folds_pkl(dataset)


def load_folds(dataset):
    with open(folds_pkl(dataset), 'rb') as fh:
        return pickle.load(fh)


if __name__ == '__main__':
    for ds in ('ncDR', 'RNAInter'):
        print('cache', build_cache(ds))
        print('folds', build_folds(ds))
