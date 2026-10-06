# -*- coding: utf-8 -*-
"""Shared data / metric / graph utilities for DLST-MDA baseline reproduction.

Benchmark: DLST-MDA (JCIM 2025) self-built dataset
  1578 miRNA x 156 drug, 8720 known associations, 5-fold, positive:negative = 1:1.
The exact author split (data/Positive.txt / Negative.txt -> train{i}.csv / test{i}.csv)
is reused so every baseline sees identical folds.
"""
import os
import random
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'repos', 'DLST-MDA-main', 'data')
PROC = os.path.join(DATA, 'processed')
RESULTS = os.path.join(ROOT, 'results')
LOGS = os.path.join(ROOT, 'logs')

DEV = os.environ.get('DLST_DEV', 'cuda:0')
N_FOLDS = 5

SEQ_VOCAB = {'A': 1, 'C': 2, 'G': 3, 'U': 4}
SEQ_LEN = 24
SMI_LEN = 100
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}


def seed_it(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seq2int(seq, max_len=SEQ_LEN):
    x = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(seq[:max_len]):
        x[i] = SEQ_VOCAB.get(ch, 0)
    return x


def smiles2int(smiles, max_len=SMI_LEN):
    x = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(smiles[:max_len]):
        x[i] = CHARISOSMISET.get(ch, 0)
    return x


# ---------------- molecular graph (identical to DLST-MDA process_data.py) ------------
def atom_features(atom):
    # re-implementation of DLST-MDA process_data.py::atom_features
    sets = (['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
             'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
             'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
             'Pt', 'Hg', 'Pb', 'Unknown'])
    def onek(x, allow):
        if x not in allow:
            raise Exception("bad")
        return list(map(lambda s: x == s, allow))
    def onek_unk(x, allow):
        if x not in allow:
            x = allow[-1]
        return list(map(lambda s: x == s, allow))
    feat = np.array(
        onek_unk(atom.GetSymbol(), sets) +
        onek(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        onek_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        onek_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        [atom.GetIsAromatic()], dtype=np.float32)
    return feat / feat.sum()


_GRAPH_CACHE = {}


def smile_to_graph(smile):
    if smile in _GRAPH_CACHE:
        return _GRAPH_CACHE[smile]
    import networkx as nx
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = [atom_features(a) for a in mol.GetAtoms()]
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = [[e1, e2] for e1, e2 in g.edges]
    out = (c_size, features, edge_index)
    _GRAPH_CACHE[smile] = out
    return out


# ---------------- metrics -------------------------------------------------------------
def compute_metrics(y_true, y_score, thr=0.5):
    from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                                 precision_score, recall_score, f1_score, matthews_corrcoef,
                                 confusion_matrix)
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).ravel()
    y_pred = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return dict(
        AUC=float(roc_auc_score(y_true, y_score)),
        AUPR=float(average_precision_score(y_true, y_score)),
        ACC=float(accuracy_score(y_true, y_pred)),
        Precision=float(precision_score(y_true, y_pred, zero_division=0)),
        Recall=float(recall_score(y_true, y_pred, zero_division=0)),
        Specificity=float(spec),
        F1=float(f1_score(y_true, y_pred, zero_division=0)),
        MCC=float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
    )


def load_fold(fold, proc=None):
    """Return (train_pairs, train_labels, test_pairs, test_labels).
    Each pair is (smiles, mirna_seq).  proc overrides the split directory."""
    import pandas as pd
    proc = proc or os.environ.get('DLST_PROC', PROC)
    tr = pd.read_csv(os.path.join(proc, f'train{fold}.csv'))
    te = pd.read_csv(os.path.join(proc, f'test{fold}.csv'))
    def pack(df):
        pairs = list(zip(df['compound_iso_smiles'].tolist(), df['target_sequence'].tolist()))
        labels = df['affinity'].astype(float).values
        return pairs, labels
    return *pack(tr), *pack(te)
