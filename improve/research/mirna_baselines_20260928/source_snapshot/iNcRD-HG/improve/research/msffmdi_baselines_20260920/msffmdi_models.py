# -*- coding: utf-8 -*-
"""Reproduce MSFFMDI's comparison baselines on ncDR / RNAInter.

Baselines (MSFFMDI paper, Sect. 3.4, Tables 3/4): LAGCN, GCNNMMA, MFIDMA, LDAMAN, DLST-MDA.
Data: MFIDMA official repository (`repos/MFIDMA-master/`).

Protocol note: the MFIDMA per-fold files all share an identical test set and contain
duplicate / train-test-overlapping positive rows, so they cannot support a real 5-CV.
We therefore build a clean, reproducible 5-fold CV over the de-duplicated positive pairs
+ an equal number of sampled negatives (the paper's stated "5-CV, positive:negative 1:1").

Usage:
  python run_msffmdi.py --model lagcn --dataset ncDR --seed 42
"""
import argparse
import csv
import os
import random
import statistics as st
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGLoader
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_add_pool, global_max_pool

HERE = os.path.dirname(os.path.abspath(__file__))
DLST = os.path.join(os.path.dirname(HERE), 'dlstmda_baselines_20260919')
sys.path.insert(0, HERE)
sys.path.insert(0, DLST)
import msffmdi_data as MD  # noqa: E402
from common import seq2int, smiles2int, smile_to_graph, SEQ_LEN, SMI_LEN  # noqa: E402

RESULTS = os.path.join(HERE, 'results')
LOGS = os.path.join(HERE, 'logs')
DEV = os.environ.get('MSFFM_DEV', 'cuda:0')
N_FOLDS = 5


def seed_it(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------------ data / features
def build_splits(dataset, seed=42):
    seqs, smis = MD.load_seq_smiles(dataset)
    names = list(seqs.keys())
    cids = list(smis.keys())
    r_idx = {q: i for i, q in enumerate(names)}
    d_idx = {s: i for i, s in enumerate(cids)}
    # unique positives across all folds
    pos = set()
    d = os.path.join(MD.REPO, dataset, 'data')
    for f in range(N_FOLDS):
        for sp in ('Train', 'Validation', 'Test'):
            pos |= set(MD._read_pairs(os.path.join(d, f'PositiveSample_{sp}{f}.csv')))
    pos = np.array([[r_idx[q], d_idx[s]] for s, q in pos], dtype=np.int64)
    # negatives: equal count from all non-associated pairs
    rng = np.random.RandomState(seed)
    n = len(pos)
    neg = np.zeros((n, 2), dtype=np.int64)
    seen = set(map(tuple, pos))
    cnt = 0
    while cnt < n:
        i = rng.randint(0, len(names))
        j = rng.randint(0, len(cids))
        if (i, j) in seen:
            continue
        seen.add((i, j))
        neg[cnt] = (i, j)
        cnt += 1
    pairs = np.concatenate([pos, neg], axis=0)
    y = np.concatenate([np.ones(n), np.zeros(n)])
    order = rng.permutation(len(pairs))
    pairs, y = pairs[order], y[order]
    chunks = np.array_split(np.arange(len(pairs)), N_FOLDS)
    folds = []
    for f in range(N_FOLDS):
        te = chunks[f]
        tr = np.concatenate([chunks[g] for g in range(N_FOLDS) if g != f])
        folds.append((tr, te))
    return dict(names=names, cids=cids, seqs=seqs, smis=smis, pairs=pairs, y=y, folds=folds)


def kmer_feat(seq, ks=(1, 2, 3)):
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
        s = v.sum()
        out.append(v / s if s > 0 else v)
    return np.concatenate(out).astype(np.float32)


def morgan(smiles, nbits=2048):
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def knn_adj(sim, k):
    n = sim.shape[0]
    a = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(sim, 0.0)
    for i in range(n):
        idx = np.argpartition(-sim[i], min(k, n - 1))[:k]
        a[i, idx] = 1.0
    a = ((a + a.T) > 0).astype(np.float32)
    np.fill_diagonal(a, 0.0)
    return a


def norm_adj(a):
    a = a + np.eye(a.shape[0], dtype=np.float32)
    deg = a.sum(1)
    dinv = np.power(deg, -0.5)
    dinv[np.isinf(dinv)] = 0
    return (dinv[:, None] * a * dinv[None, :]).astype(np.float32)


def node_features(data):
    """Return (Xr, Xd) raw node features (kmer / morgan)."""
    Xr = np.stack([kmer_feat(s) for s in data['seqs'].values()]).astype(np.float32)
    Xd = np.stack([morgan(s) for s in data['smis'].values()]).astype(np.float32)
    return Xr, Xd


# ------------------------------------------------------------------ models
class LAGCN(nn.Module):
    """Layer-attention GCN on the miRNA-drug heterogeneous similarity graph."""
    def __init__(self, fr, fd, hidden=64, layers=2):
        super().__init__()
        self.pr = nn.Linear(fr, hidden)
        self.pd = nn.Linear(fd, hidden)
        self.gcn = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(layers)])
        self.att = nn.Parameter(torch.zeros(layers + 1))

    def forward(self, xr, xd, edge_index, edge_weight):
        h = torch.cat([self.pr(xr), self.pd(xd)], 0)
        hs = [h]
        for g in self.gcn:
            h = F.relu(g(h, edge_index, edge_weight))
            hs.append(h)
        w = torch.softmax(self.att, 0)
        h = sum(wi * hi for wi, hi in zip(w, hs))
        return h[:self.nr], h[self.nr:]


class MFIDMA(nn.Module):
    """Port of the official Keras model (attribute Conv2D + SDNE behaviour Dense)."""
    def __init__(self, attr_dim=64, beh_dim=64, mi_len=25, dr_len=64):
        super().__init__()
        self.conv_mi = nn.Conv2d(1, 64, kernel_size=(4, attr_dim))
        self.conv_dr = nn.Conv2d(1, 64, kernel_size=(4, attr_dim))
        self.fc_mi = nn.Linear(64 * (mi_len - 3), 64)
        self.fc_dr = nn.Linear(64 * (dr_len - 3), 64)
        self.fc_bmi = nn.Linear(beh_dim, 64)
        self.fc_bdr = nn.Linear(beh_dim, 64)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Sequential(nn.Linear(256, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 2))

    def forward(self, mi_attr, dr_attr, mi_beh, dr_beh):
        m = F.relu(self.conv_mi(mi_attr.unsqueeze(1))).flatten(1)
        d = F.relu(self.conv_dr(dr_attr.unsqueeze(1))).flatten(1)
        m = self.drop(F.relu(self.fc_mi(m)))
        d = self.drop(F.relu(self.fc_dr(d)))
        bm = self.drop(F.relu(self.fc_bmi(mi_beh)))
        bd = self.drop(F.relu(self.fc_bdr(dr_beh)))
        return self.head(torch.cat([m, d, bm, bd], 1))


class LDAMAN(nn.Module):
    """SDNE-style autoencoder embeddings on the heterogeneous graph + MLP head
    (LDAMAN = SDNE network-behaviour features + XGBoost; here the XGBoost classifier is
    replaced by a small MLP over the same concatenated pair features, trained jointly)."""
    def __init__(self, fr, fd, hidden=64):
        super().__init__()
        self.enc_r = nn.Sequential(nn.Linear(fr, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.enc_d = nn.Sequential(nn.Linear(fd, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.dec_r = nn.Linear(hidden, fr)
        self.dec_d = nn.Linear(hidden, fd)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 128), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, xr, xd):
        zr, zd = self.enc_r(xr), self.enc_d(xd)
        return self.dec_r(zr), self.dec_d(zd), zr, zd

    def score(self, zr, zd):
        return self.head(torch.cat([zr, zd], -1)).squeeze(-1)


# DLST-MDA (official GCNNetmuti)
class DLSTMDANet(nn.Module):
    def __init__(self, n_filters=32, embed_dim=64, num_features_xd=78,
                 num_features_smile=67, num_features_xt=5, output_dim=128, dropout=0.2):
        super().__init__()
        self.smile_embed = nn.Embedding(num_features_smile, embed_dim)
        self.conv_xd_11 = nn.Conv1d(embed_dim, n_filters, 3, padding=1)
        self.conv_xd_12 = nn.Conv1d(n_filters, n_filters * 2, 3, padding=1)
        self.conv_xd_21 = nn.Conv1d(embed_dim, n_filters, 2, padding=1)
        self.conv_xd_22 = nn.Conv1d(n_filters, n_filters * 2, 2, padding=1)
        self.conv_xd_31 = nn.Conv1d(embed_dim, n_filters, 1, padding=1)
        self.conv_xd_32 = nn.Conv1d(n_filters, n_filters * 2, 1, padding=1)
        self.fc_smiles = nn.Linear(n_filters * 2, output_dim)
        self.gcnv1 = GCNConv(num_features_xd, num_features_xd * 2)
        self.gcnv2 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.fc_g1 = nn.Linear(num_features_xd * 4, 1024)
        self.fc_g2 = nn.Linear(1024, output_dim)
        self.conv_reduce_smiles = nn.Conv1d(output_dim * 3, output_dim, 1)
        self.conv_reduce_xt = nn.Conv1d(192, output_dim, 1)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_11 = nn.Conv1d(embed_dim, n_filters, 4, padding=2)
        self.conv_xt_12 = nn.Conv1d(n_filters, n_filters * 2, 4, padding=2)
        self.conv_xt_21 = nn.Conv1d(embed_dim, n_filters, 3, padding=1)
        self.conv_xt_22 = nn.Conv1d(n_filters, n_filters * 2, 3, padding=1)
        self.conv_xt_31 = nn.Conv1d(embed_dim, n_filters, 2, padding=1)
        self.conv_xt_32 = nn.Conv1d(n_filters, n_filters * 2, 2, padding=1)
        self.fc1_xt = nn.Linear(n_filters * 2, output_dim)
        self.fc1 = nn.Linear(output_dim * 2, 256)
        self.out = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        seqdrug, target = data.seqdrug, data.target
        x = self.relu(self.gcnv1(x, edge_index))
        x = self.relu(self.gcnv2(x, edge_index))
        x = global_max_pool(x, batch)
        x = self.dropout(self.fc_g1(x))
        x = self.dropout(self.fc_g2(x))
        es = self.smile_embed(seqdrug.long()).permute(0, 2, 1)
        outs = []
        for a, b in ((self.conv_xd_11, self.conv_xd_12), (self.conv_xd_21, self.conv_xd_22),
                     (self.conv_xd_31, self.conv_xd_32)):
            c = F.max_pool1d(self.relu(a(es)), 2)
            c = F.max_pool1d(self.relu(b(c)), c.size(2)).squeeze(2)
            outs.append(self.fc_smiles(c))
        cd = self.conv_reduce_smiles(torch.cat(outs, 1).unsqueeze(1).permute(0, 2, 1)).squeeze(2)
        x = cd + x
        et = self.embedding_xt(target).permute(0, 2, 1)
        outs = []
        for a, b in ((self.conv_xt_11, self.conv_xt_12), (self.conv_xt_21, self.conv_xt_22),
                     (self.conv_xt_31, self.conv_xt_32)):
            c = self.relu(a(et))
            c = F.max_pool1d(self.relu(b(c)), c.size(2)).squeeze(2)
            outs.append(c)
        ct = self.conv_reduce_xt(torch.cat(outs, 1).unsqueeze(2)).squeeze(2)
        xc = self.relu(self.fc1(torch.cat([x, ct], 1)))
        return self.out(self.dropout(xc)).squeeze(-1)


# GCNNMMA (SMiR-GCNN, copied from dlstmda_baselines gcnnma.py)
def _gcnnma_vocab(pairs, radius=2, ngram=3):
    """r-radius (WL) substructure fingerprints + n-gram words (SMiR-GCNN)."""
    from collections import defaultdict
    from rdkit import Chem
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fp_dict = defaultdict(lambda: len(fp_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    word_dict = defaultdict(lambda: len(word_dict))
    compounds, words = {}, {}

    def create_atoms(mol):
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        for a in mol.GetAromaticAtoms():
            atoms[a.GetIdx()] = (atoms[a.GetIdx()], 'aromatic')
        return np.array([atom_dict[a] for a in atoms])

    def create_ijbonddict(mol):
        d = defaultdict(list)
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond = bond_dict[str(b.GetBondType())]
            d[i].append((j, bond)); d[j].append((i, bond))
        return d

    def extract(atoms, i_jedge):
        n = len(atoms)
        if n == 1 or radius == 0:
            return atoms
        nodes = {i: atoms[i] for i in range(n)}
        edges = {i: list(v) for i, v in i_jedge.items()}
        for _ in range(radius):
            new_nodes = {}
            for i, j_edge in edges.items():
                neighbors = [(nodes[j], e) for j, e in j_edge]
                new_nodes[i] = fp_dict[(nodes[i], tuple(sorted(neighbors)))]
            new_edges = defaultdict(list)
            for i, j_edge in edges.items():
                for j, e in j_edge:
                    both = tuple(sorted((new_nodes[i], new_nodes[j])))
                    new_edges[i].append((j, edge_dict[(both, e)]))
            nodes, edges = new_nodes, new_edges
        out = np.zeros(n, dtype=np.int64)
        for i in range(n):
            out[i] = nodes.get(i, atoms[i])
        return out

    for smi, seq in pairs:
        if smi not in compounds:
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            fp = extract(create_atoms(mol), create_ijbonddict(mol))
            adj = np.array(Chem.GetAdjacencyMatrix(mol), dtype=np.float32)
            compounds[smi] = (torch.tensor(fp, dtype=torch.long), torch.tensor(adj))
        if seq not in words:
            ss = '-' + seq + '='
            words[seq] = [word_dict[ss[i:i + ngram]] for i in range(len(ss) - ngram + 1)]
    return compounds, words, len(fp_dict), len(word_dict)


class GCNNMMA(nn.Module):
    def __init__(self, n_fp, n_word, dim=10, layer_gnn=3, window=11, layer_cnn=3, layer_out=3):
        super().__init__()
        self.embed_fp = nn.Embedding(n_fp, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(1, 1, 2 * window + 1, padding=window)
                                    for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2 * dim, 2 * dim) for _ in range(layer_out)])
        self.W_interaction = nn.Linear(2 * dim, 2)

    def forward(self, fp, adj, mask, words):
        x = self.embed_fp(fp) * mask.unsqueeze(-1)
        for w in self.W_gnn:
            x = (x + torch.bmm(adj, torch.relu(w(x)))) * mask.unsqueeze(-1)
        compound = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        xs = self.embed_word(words).permute(0, 2, 1).unsqueeze(1)
        for w in self.W_cnn:
            xs = torch.relu(w(xs))
        xs = xs.squeeze(1).permute(0, 2, 1)
        h = torch.relu(self.W_attention(compound))
        hs = torch.relu(self.W_attention(xs))
        att = torch.tanh(torch.bmm(h.unsqueeze(1), hs.permute(0, 2, 1)).squeeze(1))
        protein = (att.unsqueeze(-1) * hs).mean(1)
        cat = torch.cat([compound, protein], 1)
        for w in self.W_out:
            cat = torch.relu(w(cat))
        return self.W_interaction(cat)
