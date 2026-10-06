# -*- coding: utf-8 -*-
"""DLST-MDA port of iNcRD-HG's graph-context pair features (PairFeatureEncoder).

iNcRD-HG's strongest reference = TabICLv2 on a 36-column graph-context feature bank
(improve/graph_tabm_modules.py::PairFeatureEncoder) fused with an evidence tier and an
anchor-shrinkage post-hoc ensemble.  The DLST-MDA benchmark has no expression / evidence /
gene-pathway data, so only the **graph-context tier** is portable.  This module rebuilds
those columns for the 1578x156 miRNA-drug benchmark:

  node features : miRNA = 2/3-mer frequencies; drug = Morgan fingerprint (r=2, 2048 bit)
  rna graph sr  : top-k cosine kNN over miRNA k-mer features (binary, no self loops)
  drug graph sd : top-k Tanimoto kNN over Morgan fingerprints (binary, no self loops)
  bridge        : 0 (single-task benchmark; no lncRNA->miRNA cross link)

Columns returned mirror PairFeatureEncoder.transform (bridge/bridge_norm are 0).
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BL = '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/dlstmda_baselines_20260919'
PROC = os.path.join(BL, 'repos', 'DLST-MDA-main', 'data', 'processed')

KNN_R = 10
KNN_D = 10


def load_pairs(proc=PROC):
    """Return (mirna_seq list, drug_smiles list, fold_data, all_pairs)."""
    import csv
    folds = {}
    all_pairs = {}
    for f in range(5):
        for split in ('train', 'test'):
            rows = list(csv.DictReader(open(os.path.join(proc, f'{split}{f}.csv'))))
            obj = ([(r['compound_iso_smiles'], r['target_sequence']) for r in rows],
                   np.array([float(r['affinity']) for r in rows]))
            folds[(f, split)] = obj
            for p, y in zip(*obj):
                all_pairs[p] = int(y)
    mirnas = list(dict.fromkeys(q for _, q in all_pairs))
    drugs = list(dict.fromkeys(s for s, _ in all_pairs))
    return mirnas, drugs, folds, all_pairs


def kmer_features(seq, ks=(2, 3)):
    out = []
    for k in ks:
        v = np.zeros(4 ** k, dtype=np.float32)
        for i in range(len(seq) - k + 1):
            code, ok = 0, True
            for ch in seq[i:i + k]:
                b = {'A': 0, 'C': 1, 'G': 2, 'U': 3}.get(ch)
                if b is None:
                    ok = False
                    break
                code = code * 4 + b
            if ok:
                v[code] += 1
        n = v.sum()
        out.append(v / n if n > 0 else v)
    return np.concatenate(out).astype(np.float32)


def morgan(smiles, nbits=2048):
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def knn_adjacency(sim, k):
    n = sim.shape[0]
    a = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(sim, 0.0)
    for i in range(n):
        idx = np.argpartition(-sim[i], min(k, n - 1))[:k]
        a[i, idx] = 1.0
    a = ((a + a.T) > 0).astype(np.float32)
    np.fill_diagonal(a, 0.0)
    return a


class GraphPairEncoder:
    """Re-implementation of iNcRD-HG PairFeatureEncoder for a single RNA-drug task."""

    def __init__(self, xr, xd, sr, sd):
        self.xr, self.xd = xr.astype(np.float32), xd.astype(np.float32)
        self.sr, self.sd = sr.astype(np.float32), sd.astype(np.float32)
        self.nr, self.nd = self.xr.shape[0], self.xd.shape[0]
        self.ur, self.ud = self.unit(self.xr), self.unit(self.xd)
        # side degree columns (MiMi, DrugDrug)
        self.deg_r_side = self.sr.sum(1)
        self.deg_d_side = self.sd.sum(0)

    @staticmethod
    def unit(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

    def transform(self, pairs, support_positive):
        pairs = np.asarray(pairs, dtype=np.int64)
        support_positive = np.asarray(support_positive, dtype=np.int64)
        a = np.zeros((self.nr, self.nd), dtype=np.float32)
        if support_positive.size:
            a[support_positive[0], support_positive[1]] = 1.0
        r, d = pairs
        if np.any(a[r, d]):
            raise ValueError('A query positive is visible to its own feature encoder')
        dr, dd = a.sum(1), a.sum(0)
        sr_a, a_sd = self.sr @ a, a @ self.sd
        gram = a.T @ a
        a3 = a @ gram
        a5 = a3 @ gram
        u, sig, vt = np.linalg.svd(a, full_matrices=False)
        rank = min(16, len(sig))
        factor = (u[:, :rank] * sig[:rank]) @ vt[:rank]
        r_neighbor = (a @ dd) / np.maximum(dr, 1)
        d_neighbor = (a.T @ dr) / np.maximum(dd, 1)
        attribute_d = self.unit(a @ self.ud)
        attribute_r = self.unit(a.T @ self.ur)
        expr_support = self.ur @ self.ur.T
        drug_support = self.ud @ self.ud.T
        np.fill_diagonal(expr_support, 0.0)
        np.fill_diagonal(drug_support, 0.0)
        cols = {
            'degree_r': dr[r], 'degree_d': dd[d], 'degree_product': dr[r] * dd[d],
            'collab_r': sr_a[r, d], 'collab_d': a_sd[r, d],
            'collab_r_mean': sr_a[r, d] / np.maximum(self.sr.sum(1)[r], 1),
            'collab_d_mean': a_sd[r, d] / np.maximum(self.sd.sum(0)[d], 1),
            'collab_both': (sr_a @ self.sd)[r, d],
            'path3': a3[r, d], 'path3_norm': a3[r, d] / np.maximum(dr[r] * dd[d], 1),
            'path5': a5[r, d], 'path5_norm': a5[r, d] / np.maximum(dr[r] * dd[d], 1),
            'svd16': factor[r, d], 'neighbor_degree_r': r_neighbor[r],
            'neighbor_degree_d': d_neighbor[d],
            'bridge': np.zeros(len(r), dtype=np.float32),
            'bridge_norm': np.zeros(len(r), dtype=np.float32),
            'rna_similarity_degree': (self.sr @ dr)[r],
            'drug_similarity_degree': (self.sd @ dd)[d],
            'collab_r2': (self.sr @ sr_a)[r, d],
            'collab_d2': (a_sd @ self.sd)[r, d],
            'rna_attribute_support_sum': (expr_support @ a)[r, d],
            'drug_attribute_support_sum': (a @ drug_support)[r, d],
            'drug_attribute_profile_cos': (attribute_d[r] * self.ud[d]).sum(1),
            'rna_attribute_profile_cos': (attribute_r[d] * self.ur[r]).sum(1),
            'rna_feature_mean': self.xr.mean(1)[r], 'rna_feature_std': self.xr.std(1)[r],
            'rna_feature_norm': np.linalg.norm(self.xr, axis=1)[r],
            'drug_feature_mean': self.xd.mean(1)[d], 'drug_feature_std': self.xd.std(1)[d],
            'drug_feature_norm': np.linalg.norm(self.xd, axis=1)[d],
            'degree_MiMi': self.deg_r_side[r], 'degree_DrugDrug': self.deg_d_side[d],
        }
        frame = pd.DataFrame(cols, dtype=np.float32)
        if not np.isfinite(frame.values).all():
            raise ValueError('Non-finite pair features')
        return frame


def crossfit_features(encoder, pairs, labels, evaluation_pairs, folds=5, seed=20260910, eval_support=None):
    from sklearn.model_selection import StratifiedKFold
    features = None
    names = None
    for fold, (fit, query) in enumerate(
            StratifiedKFold(folds, shuffle=True, random_state=seed).split(pairs.T, labels)):
        support = pairs[:, fit[labels[fit] == 1]]
        frame = encoder.transform(pairs[:, query], support)
        if features is None:
            features = np.zeros((len(labels), frame.shape[1]), dtype=np.float32)
            names = list(frame.columns)
        features[query] = frame.values
    if eval_support is None:
        eval_support = pairs[:, labels == 1]
    evaluation = encoder.transform(evaluation_pairs, eval_support)
    return features, evaluation.values.astype(np.float32), names


def encoder_from_arrays(xr, xd):
    """Build the pair encoder from precomputed node features (numpy only)."""
    un = xr / np.maximum(np.linalg.norm(xr, axis=1, keepdims=True), 1e-8)
    sr = knn_adjacency((un @ un.T).astype(np.float32), KNN_R)
    xdn = xd / np.maximum(np.linalg.norm(xd, axis=1, keepdims=True), 1e-8)
    sd = knn_adjacency((xdn @ xdn.T).astype(np.float32), KNN_D)
    return GraphPairEncoder(xr, xd, sr, sd)


def build_encoder(mirnas, drugs):
    """Needs rdkit; use build_cache + encoder_from_arrays instead when rdkit is absent."""
    xr = np.stack([kmer_features(q) for q in mirnas])
    xd = np.stack([morgan(s) for s in drugs])
    return encoder_from_arrays(xr, xd)


CACHE = os.path.join(HERE, 'graph_cache.npz')


def build_cache(path=CACHE):
    """Precompute node features in the rdkit environment and persist them."""
    mirnas, drugs, folds, allp = load_pairs()
    xr = np.stack([kmer_features(q) for q in mirnas]).astype(np.float32)
    xd = np.stack([morgan(s) for s in drugs]).astype(np.float32)
    np.savez_compressed(path, xr=xr, xd=xd,
                        mirnas=np.array(mirnas, dtype=object), drugs=np.array(drugs, dtype=object))
    return path


def load_cache(path=CACHE):
    d = np.load(path, allow_pickle=True)
    return d['xr'].astype(np.float32), d['xd'].astype(np.float32), list(d['mirnas']), list(d['drugs'])


def fold_arrays(mirnas, drugs, folds, fold, encoder=None, drop_dup=True):
    ridx = {q: i for i, q in enumerate(mirnas)}
    didx = {s: i for i, s in enumerate(drugs)}
    if encoder is None:
        encoder = build_encoder(mirnas, drugs)
    def dedup(pairs, labels):
        # Node identity is the molecular entity (unique sequence / canonical SMILES), so a
        # few author rows collapse; on a positive/negative collision keep the positive
        # (the same molecule is a known association).
        seen = {}
        order = []
        for (s, q), y in zip(pairs, labels):
            key = (ridx[q], didx[s])
            if key not in seen:
                seen[key] = int(y)
                order.append(key)
            else:
                seen[key] = max(seen[key], int(y))
        return (np.array(order, dtype=np.int64).T,
                np.array([seen[k] for k in order], dtype=np.int64))

    ptr, tr_y = dedup(*folds[(fold, 'train')])
    if drop_dup:
        pte, te_y = dedup(*folds[(fold, 'test')])
        # remove test pairs whose (entity, entity) key is already in train, so the evaluation
        # positive support is disjoint at the molecular-entity level (few duplicate-sequence rows)
        train_keys = set(map(tuple, ptr.T))
        keep = np.array([tuple(k) not in train_keys for k in pte.T])
        dropped = int((~keep).sum())
        pte, te_y = pte[:, keep], te_y[keep]
        if dropped:
            print(f'  fold {fold}: dropped {dropped} test pairs duplicated in train (entity-level)', flush=True)
        eval_support = None
    else:
        # keep-all (author-row protocol, as DLMVF/DLST-MDA): score every author test row, duplicates included.
        # Train positives that coincide with a test key (entity-level) are masked out of the evaluation
        # support so no query positive is visible to its own feature encoder (still leak-free).
        te_pairs, te_lab = folds[(fold, 'test')]
        pte = np.array([(ridx[q], didx[s]) for (s, q) in te_pairs], dtype=np.int64).T
        te_y = np.asarray(te_lab, dtype=np.int64)
        test_keys = set(map(tuple, pte.T))
        pos = ptr[:, tr_y == 1]
        mask = np.array([tuple(k) not in test_keys for k in pos.T])
        eval_support = pos[:, mask]
        print(f'  fold {fold}: keep-all {pte.shape[1]} test rows; masked {int((~mask).sum())} train positives from eval support', flush=True)
    xtr, xte, names = crossfit_features(encoder, ptr, tr_y, pte, eval_support=eval_support)
    return dict(x_train=xtr, y_train=tr_y, pairs_train=ptr,
                x_test=xte, y_test=te_y, pairs_test=pte, names=names)
