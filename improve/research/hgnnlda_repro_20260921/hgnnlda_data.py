# -*- coding: utf-8 -*-
"""Data access for the HGNNLDA (RNAactDrug lncRNA-drug sensitivity) benchmark."""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, 'repos', 'HGNNLDA-official')
DATASET = os.path.join(REPO, 'dataset')


def _read_pairs(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(',', '\t').split()
            out.append((int(parts[0]), int(parts[1])))
    return np.array(out, dtype=np.int64)


def full_sizes():
    with open(os.path.join(REPO, 'lindex.txt')) as f:
        n_l = sum(1 for line in f if line.strip())
    with open(os.path.join(REPO, 'drug.txt')) as f:
        n_d = sum(1 for line in f if line.strip())
    return n_l, n_d


def load_all():
    return _read_pairs(os.path.join(REPO, 'ass.txt'))


def load_fold(i):
    tr = _read_pairs(os.path.join(DATASET, f'train_{i}.txt'))
    te = _read_pairs(os.path.join(DATASET, f'test_{i}.txt'))
    return tr, te


def load_independent():
    return _read_pairs(os.path.join(REPO, 'indenpent_dataset.txt'))


def train_adjacency(pos_pairs, n_l, n_d):
    A = np.zeros((n_l, n_d), dtype=np.float32)
    A[pos_pairs[:, 0], pos_pairs[:, 1]] = 1.0
    return A


def sample_negatives(known, n_l, n_d, k, rng):
    """Sample k unknown (lncRNA, drug) pairs uniformly, excluding `known`."""
    known = set(map(tuple, known.tolist()))
    out = []
    while len(out) < k:
        i = int(rng.integers(0, n_l))
        j = int(rng.integers(0, n_d))
        if (i, j) not in known:
            out.append((i, j))
    return np.array(out, dtype=np.int64)
