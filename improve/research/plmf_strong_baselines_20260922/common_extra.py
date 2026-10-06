# -*- coding: utf-8 -*-
"""Shared utilities for the *additional* strong-miRNA baselines on PLMF-MDA.

Baselines added here (all on the PLMF-MDA MDRdataset / MDSdataset, paper 5-fold
protocol, seed 42, test-best-epoch -- identical to `../plmf_baselines_20260920`):
  * DLST-MDA  (official `GCNNetmuti`, code/cnn_gcnmulti.py)
  * DLMVF     (official `DLMVF`, model.py)
  * SubKNet   (official Model / RW_layer / subgraph extraction)

Everything reuses the exact folds / featurisers of the existing PLMF baseline
suite so the new numbers are directly comparable:

  `../plmf_baselines_20260920/plmf_data.py`  -> load_fold_5cv / load_dataset
  `../dlstmda_baselines_20260919/common.py`  -> seq2int / smiles2int /
                                                 smile_to_graph / compute_metrics
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
DLST = os.path.join(BASE, 'dlstmda_baselines_20260919')
PLMFB = os.path.join(BASE, 'plmf_baselines_20260920')
STRONG = os.path.join(BASE, 'plmf_strongest_20260920')

for _p in (DLST, PLMFB, STRONG, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (  # noqa: E402
    DEV, seed_it, seq2int, smiles2int, smile_to_graph, compute_metrics,
    SEQ_LEN, SMI_LEN, CHARISOSMISET,
)
import plmf_data  # noqa: E402

RESULTS = os.path.join(HERE, 'results')
LOGS = os.path.join(HERE, 'logs')
DATA = os.path.join(HERE, 'data')
N_FOLDS = 5

METHOD_NAME = {'dlstmda': 'DLST-MDA', 'dlmvf': 'DLMVF', 'subknet': 'SubKNet'}


def get_fold(dataset, split, fold, seed):
    """Same folds as the PLMF baseline suite (5cv primary / single 80-20)."""
    if split == '5cv':
        return plmf_data.load_fold_5cv(dataset, fold, seed)
    return plmf_data.load_single_split(dataset, seed)


def unique_entities(dataset):
    """Unique miRNA sequences / drug SMILES and their index maps."""
    seqs, smiles, _, _ = plmf_data.load_dataset(dataset)
    mirnas = list(dict.fromkeys(seqs))
    drugs = list(dict.fromkeys(smiles))
    return mirnas, drugs


def graph_cache(dataset):
    """k-mer (80-d) miRNA features + Morgan (2048-bit) drug features, cached by
    the `plmf_strongest_20260920` pipeline."""
    import numpy as np
    d = np.load(os.path.join(STRONG, f'graph_cache_{dataset}.npz'), allow_pickle=True)
    return (d['xr'].astype(np.float32), d['xd'].astype(np.float32),
            list(d['mirnas']), list(d['drugs']))


def csv_fields():
    return ['dataset', 'split', 'method', 'fold', 'seed', 'auc', 'aupr', 'acc',
            'precision', 'recall', 'specificity', 'f1', 'mcc', 'best_epoch',
            'last_auc', 'last_aupr']


def row_from_metrics(dataset, split, method, fold, seed, best, last):
    return {
        'dataset': dataset, 'split': split, 'method': method, 'fold': fold,
        'seed': seed,
        'auc': best['AUC'], 'aupr': best['AUPR'], 'acc': best['ACC'],
        'precision': best['Precision'], 'recall': best['Recall'],
        'specificity': best['Specificity'], 'f1': best['F1'], 'mcc': best['MCC'],
        'best_epoch': best['epoch'], 'last_auc': last['AUC'], 'last_aupr': last['AUPR'],
    }


def summarize_rows(rows, tag=''):
    import statistics as st
    print(f'\n=== {tag} ({len(rows)} folds) ===')
    out = {}
    for k in ['auc', 'aupr', 'acc', 'f1', 'mcc']:
        v = [r[k] for r in rows]
        sd = st.stdev(v) if len(v) > 1 else 0.0
        out[k] = (st.mean(v), sd)
        print(f'  {k}: {st.mean(v):.4f} +/- {sd:.4f}')
    return out
