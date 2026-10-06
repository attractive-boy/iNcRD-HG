# -*- coding: utf-8 -*-
"""Data module for reproducing MSFFMDI's comparison baselines on ncDR / RNAInter.

Datasets and 5-fold splits come from the MFIDMA official repository
(`repos/MFIDMA-master/`), which MSFFMDI explicitly reused
("based on the work of Guan et al. on MFIDMA ... organized in their study").

Per fold f (0..4) the authors provide:
  PositiveSample_Train{f}.csv / Validation{f} / Test{f}   (drug_cid, mirna_name)
  NegativeSample_Train{f}.csv / Validation{f} / Test{f}
ncDR:     train 3119 / val 446 / test 892  (pos and neg each), total 4457
RNAInter: train 4592 / val 574 / test 574  (pos and neg each), total 5740
"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, 'repos', 'MFIDMA-master')

DATASETS = {
    'ncDR': dict(seq='ncDR/ncDRAllMiSequence.csv', smi='ncDR/ncDRAllDrugSmiles.csv'),
    'RNAInter': dict(seq='RNAInter/RNAInterAllMiSequence.csv',
                     smi='RNAInter/RNAInterAllDrugCanonicalSMILES.csv'),
}
N_FOLDS = 5


def _read_pairs(path):
    out = []
    with open(path, newline='') as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                out.append((str(row[0]).strip(), str(row[1]).strip()))
    return out


def load_seq_smiles(dataset):
    cfg = DATASETS[dataset]
    seqs, smis = {}, {}
    with open(os.path.join(REPO, cfg['seq']), newline='') as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                seqs[str(row[0]).strip()] = str(row[1]).strip().upper().replace('T', 'U')
    with open(os.path.join(REPO, cfg['smi']), newline='') as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                smis[str(row[0]).strip()] = str(row[1]).strip()
    return seqs, smis


def load_fold(dataset, fold, use_validation=False):
    """Return (train_pairs, train_y, test_pairs, test_y).
    pairs = (smiles, mirna_sequence); positives then negatives.
    If use_validation, validation pairs are appended to train."""
    d = os.path.join(REPO, dataset, 'data')
    def rd(split, label):
        pairs = _read_pairs(os.path.join(d, f'{"Positive" if label else "Negative"}Sample_{split}{fold}.csv'))
        return pairs, np.full(len(pairs), float(label))
    tr_p, tr_y = rd('Train', 1)
    tr_n, tr_y0 = rd('Train', 0)
    te_p, te_y1 = rd('Test', 1)
    te_n, te_y0 = rd('Test', 0)
    if use_validation:
        va_p, va_y1 = rd('Validation', 1)
        va_n, va_y0 = rd('Validation', 0)
        tr_p = tr_p + va_p + tr_n + va_n
        tr_y = np.concatenate([tr_y, va_y1, tr_y0, va_y0])
    else:
        tr_p = tr_p + tr_n
        tr_y = np.concatenate([tr_y, tr_y0])
    te_pairs = te_p + te_n
    te_y = np.concatenate([te_y1, te_y0])
    seqs, smis = load_seq_smiles(dataset)
    def to_pairs(ps):
        return [(smis[s], seqs[q]) for s, q in ps]
    return to_pairs(tr_p), tr_y, to_pairs(te_pairs), te_y


def association_matrix(dataset, folds=None, use_validation=True):
    """Full association matrix over the dataset's miRNA/drug vocabulary (union of
    train+test for the given folds; defaults to all folds = the paper's known pairs)."""
    seqs, smis = load_seq_smiles(dataset)
    names = list(seqs.keys())
    cids = list(smis.keys())
    r_idx = {q: i for i, q in enumerate(names)}
    d_idx = {s: i for i, s in enumerate(cids)}
    A = np.zeros((len(names), len(cids)), dtype=np.float32)
    fl = range(N_FOLDS) if folds is None else folds
    for f in fl:
        d = os.path.join(REPO, dataset, 'data')
        for split in ('Train', 'Test') + (('Validation',) if use_validation else ()):
            for label, fname in ((1, 'Positive'), (0, 'Negative')):
                for s, q in _read_pairs(os.path.join(d, f'{fname}Sample_{split}{f}.csv')):
                    if label == 1:
                        A[r_idx[q], d_idx[s]] = 1.0
    return names, cids, A


def compute_metrics(y, score, thr=0.5):
    from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                                 recall_score, confusion_matrix)
    y = np.asarray(y).astype(int).ravel()
    score = np.asarray(score).ravel()
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(AUC=float(roc_auc_score(y, score)),
                AUPR=float(average_precision_score(y, score)),
                ACC=float(accuracy_score(y, pred)),
                Sen=float(recall_score(y, pred, zero_division=0)),
                Spec=float(tn / (tn + fp) if (tn + fp) else 0.0))


if __name__ == '__main__':
    for ds in DATASETS:
        seqs, smis = load_seq_smiles(ds)
        tr, ty, te, ey = load_fold(ds, 0)
        print(ds, 'miRNAs', len(seqs), 'drugs', len(smis),
              'train', len(tr), int(ty.sum()), 'test', len(te), int(ey.sum()))
        names, cids, A = association_matrix(ds)
        print('   assoc matrix', A.shape, 'positives', int(A.sum()))
