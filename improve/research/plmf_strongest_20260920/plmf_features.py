# -*- coding: utf-8 -*-
"""Node features + fold adapters to run iNcRD-HG's strongest *portable* reference on
PLMF-MDA's MDR/MDS benchmark.

Reuses the graph-context pair encoder of
`../dlstmda_strongest_20260920/features.py` (the portable tier of the iNcRD-HG
reference).  Only the data source changes: PLMF-MDA MDR/MDS (miRNA sequences +
drug SMILES + associations), 5cv or released 80/20 split from `../plmf_baselines_20260920`.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
STRONG = os.path.join(BASE, 'dlstmda_strongest_20260920')
PLMFB = os.path.join(BASE, 'plmf_baselines_20260920')
sys.path.insert(0, STRONG)
sys.path.insert(0, PLMFB)
sys.path.insert(0, HERE)

import features as F  # noqa: E402
import plmf_data  # noqa: E402


def load_plmf(dataset, split, seed):
    """Return (mirnas, drugs, fold_data) where fold_data[(f,'train'|'test')] = (pairs, y).

    The fold data is pickled by `build_folds_cache` (run in the rdkit env) so the TabICL
    venv does not need openpyxl/pandas-excel.
    """
    pkl = folds_cache_path(dataset, split, seed)
    if os.path.exists(pkl):
        import pickle
        with open(pkl, 'rb') as fh:
            return pickle.load(fh)
    seqs, smiles, pos, neg = plmf_data.load_dataset(dataset)
    mirnas = list(dict.fromkeys(seqs))
    drugs = list(dict.fromkeys(smiles))
    folds = {}
    if split == '5cv':
        for f in range(5):
            tr, ty, te, ey = plmf_data.load_fold_5cv(dataset, f, seed)
            folds[(f, 'train')] = (tr, ty)
            folds[(f, 'test')] = (te, ey)
    else:
        tr, ty, te, ey = plmf_data.load_single_split(dataset, seed)
        folds[(0, 'train')] = (tr, ty)
        folds[(0, 'test')] = (te, ey)
    return mirnas, drugs, folds


def folds_cache_path(dataset, split, seed):
    return os.path.join(HERE, f'folds_{dataset}_{split}_{seed}.pkl')


def build_folds_cache():
    import pickle
    out = []
    for dataset in ('MDR', 'MDS'):
        for split, seeds in (('5cv', [42]), ('single', [0, 1, 2, 3, 4])):
            for seed in seeds:
                pkl = folds_cache_path(dataset, split, seed)
                if os.path.exists(pkl):
                    continue
                data = load_plmf(dataset, split, seed)
                with open(pkl, 'wb') as fh:
                    pickle.dump(data, fh)
                out.append(pkl)
    return out


def cache_path(dataset):
    return os.path.join(HERE, f'graph_cache_{dataset}.npz')


def build_cache(dataset):
    """Needs rdkit -> run in the ncRNAdrug env."""
    seqs, smiles, _, _ = plmf_data.load_dataset(dataset)
    mirnas = list(dict.fromkeys(seqs))
    drugs = list(dict.fromkeys(smiles))
    xr = np.stack([F.kmer_features(q) for q in mirnas]).astype(np.float32)
    xd = np.stack([F.morgan(s) for s in drugs]).astype(np.float32)
    np.savez_compressed(cache_path(dataset), xr=xr, xd=xd,
                        mirnas=np.array(mirnas, dtype=object),
                        drugs=np.array(drugs, dtype=object))
    return cache_path(dataset)


def load_cache(dataset):
    d = np.load(cache_path(dataset), allow_pickle=True)
    return (d['xr'].astype(np.float32), d['xd'].astype(np.float32),
            list(d['mirnas']), list(d['drugs']))


if __name__ == '__main__':
    for ds in ('MDR', 'MDS'):
        print('built', build_cache(ds))
