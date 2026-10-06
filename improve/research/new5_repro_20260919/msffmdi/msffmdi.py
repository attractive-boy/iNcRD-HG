#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSFFMDI -- independent REIMPLEMENTATION from the published abstract/method
description of:

  Lei et al., "Predicting miRNA-Drug Interactions Based on Multi-source Feature
  Fusion of Heterogeneous Network", Interdisciplinary Sciences: Computational
  Life Sciences, 2025. DOI 10.1007/s12539-025-00775-7

THERE IS NO OFFICIAL CODE AND THE FULL TEXT IS PAYWALLED. This file is a
from-the-abstract reimplementation, NOT a reproduction of the authors'
pipeline and NOT a reproduction of the reported numbers. Every number produced
by this script comes from an actual run on the shared DLST-MDA benchmark.

Design summary (see REPRO_MSFFMDI.md for full justification):

  Channel 1 (attribute):
    miRNA -> concatenated 1/2/3-mer frequency vector + gensim Word2Vec
             (skip-gram) mean-pooled k-mer token embedding.
    drug  -> (a) GIN over the RDKit molecular graph (trainable);
             (b) "mol2vec" substitute: gensim Word2Vec trained on SMILES token
                 sequences (the original pretrained mol2vec model is unavailable
                 offline).

  Channel 2 (topological):
    Heterogeneous network built from Gaussian-kernel (GIP) miRNA-miRNA and
    drug-drug similarities (top-k sparsified) plus the known miRNA-drug
    association edges. Node initial features = channel-1 attribute embeddings.
    GATConv updates node embeddings; a multi-scale 1D CNN over each node's
    feature map extracts the topological representation.

  Fusion: concatenate attribute + topological node features, per-fold PCA
    (fit on training-fold nodes) to reduce, then an MLP pair classifier with
    sigmoid output. BCE loss, Adam(lr=5e-4, weight_decay=5e-4), dropout=0.5.

Fold protocol is EXACTLY the DLST-MDA one: Positive.txt / Negative.txt contain
integer indices into np.where(A==1) / np.where(A!=1). Fold i test = fold-i
positives + fold-i negatives; train = the other four folds (pos & neg).
"""

import os
import sys
import json
import time
import random
import argparse
from collections import Counter

BASE = "/data1/licheng/incrd-hg/iNcRD-HG/improve/research/new5_repro_20260919"
# gensim (and only gensim) is installed into an isolated --target dir so the
# shared conda env is untouched. Appending (not prepending) keeps the env's
# numpy/scipy authoritative.
sys.path.append(os.path.join(BASE, "venvs", "msffmdi_site"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, GATConv, global_add_pool
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    matthews_corrcoef, precision_score, recall_score,
)
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from gensim.models import Word2Vec

DATA = os.path.join(BASE, "repos", "DLST-MDA-main", "data")
CACHE = os.path.join(BASE, "msffmdi", "cache")
OUT = os.path.join(BASE, "msffmdi")


# --------------------------------------------------------------------------- #
# utilities
# --------------------------------------------------------------------------- #
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg, fh=None):
    line = str(msg)
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


# --------------------------------------------------------------------------- #
# data loading + fold construction (exact DLST-MDA protocol)
# --------------------------------------------------------------------------- #
def load_data():
    A_df = pd.read_excel(os.path.join(DATA, "miRNA_drug_matrix.xlsx"), index_col=0)
    seq_df = pd.read_excel(os.path.join(DATA, "miRNA_sequences.xlsx"))
    drug_df = pd.read_excel(os.path.join(DATA, "drug_id_smiles.xlsx"))
    A = A_df.values.astype(int)                                  # (1578, 156)
    mi_names = A_df.index.astype(str).tolist()
    dr_names = A_df.columns.astype(str).tolist()
    sequences = seq_df["Sequence"].astype(str).tolist()
    smiles = drug_df["smiles"].astype(str).tolist()
    assert len(sequences) == A.shape[0] and len(smiles) == A.shape[1]
    assert seq_df.iloc[:, 0].astype(str).tolist() == mi_names
    assert drug_df.iloc[:, 0].astype(str).tolist() == dr_names
    return A, sequences, smiles, mi_names, dr_names


def build_folds(A):
    """Return list of 5 (train_pairs, train_y, test_pairs, test_y).

    Positive.txt/Negative.txt are lists of indices into the row-major lists of
    positive / negative matrix cells (this is how DLST-MDA's process_data.py
    consumes them)."""
    pos = json.load(open(os.path.join(DATA, "Positive.txt")))
    neg = json.load(open(os.path.join(DATA, "Negative.txt")))
    po_rows, po_cols = np.where(A == 1)
    ne_rows, ne_cols = np.where(A != 1)

    folds = []
    for i in range(5):
        tr_idx_p = np.concatenate([pos[j] for j in range(5) if j != i])
        tr_idx_n = np.concatenate([neg[j] for j in range(5) if j != i])
        te_idx_p = np.asarray(pos[i])
        te_idx_n = np.asarray(neg[i])

        def pairs(pr, pc, ii):
            return list(zip(pr[ii].tolist(), pc[ii].tolist()))

        tr = pairs(po_rows, po_cols, tr_idx_p) + pairs(ne_rows, ne_cols, tr_idx_n)
        te = pairs(po_rows, po_cols, te_idx_p) + pairs(ne_rows, ne_cols, te_idx_n)
        ytr = np.array([1] * len(tr_idx_p) + [0] * len(tr_idx_n), dtype=np.float32)
        yte = np.array([1] * len(te_idx_p) + [0] * len(te_idx_n), dtype=np.float32)
        folds.append((tr, ytr, te, yte))
    return folds


# --------------------------------------------------------------------------- #
# channel 1a -- miRNA attributes: k-mer frequency + word2vec
# --------------------------------------------------------------------------- #
def kmer_frequency_vector(seq, ks=(1, 2, 3), alphabet="ACGU"):
    """Concatenated 1/2/3-mer relative-frequency vector (84-d for ACGU)."""
    out = []
    for k in ks:
        vocab = [""]
        for _ in range(k):
            vocab = [p + c for p in vocab for c in alphabet]
        vocab = [v for v in vocab if v != ""]
        idx = {v: i for i, v in enumerate(vocab)}
        vec = np.zeros(len(vocab), dtype=np.float32)
        n = 0
        for i in range(len(seq) - k + 1):
            tok = seq[i:i + k]
            if tok in idx:
                vec[idx[tok]] += 1.0
                n += 1
        if n > 0:
            vec /= n
        out.append(vec)
    return np.concatenate(out)


def mirna_attributes(sequences, dim=64, seed=42, cache=True, fh=None):
    """Concatenated k-mer frequency vector + mean-pooled Word2Vec k-mer tokens."""
    cache_f = os.path.join(CACHE, f"mirna_attr_dim{dim}_seed{seed}.npz")
    if cache and os.path.isfile(cache_f):
        z = np.load(cache_f)
        log(f"[attr/miRNA] cache hit {cache_f}", fh)
        return z["freq"], z["w2v"], z["attr"]

    # k-mer frequency part (concatenate 1/2/3-mer frequency vectors)
    freq = np.stack([kmer_frequency_vector(s) for s in sequences])      # (N,84)
    freq = freq / (np.linalg.norm(freq, axis=1, keepdims=True) + 1e-8)

    # word2vec corpus: one "document" per miRNA = its 1/2/3-mer token list
    corpus = []
    for s in sequences:
        toks = []
        for k in (1, 2, 3):
            toks += [s[i:i + k] for i in range(len(s) - k + 1)]
        corpus.append(toks)
    w2v = Word2Vec(sentences=corpus, vector_size=dim, window=5, min_count=1,
                   sg=1, workers=4, epochs=50, seed=seed, negative=5)
    w2v_emb = np.stack([
        np.mean([w2v.wv[t] for t in toks], axis=0) for toks in corpus
    ]).astype(np.float32)                                              # (N,dim)

    attr = np.concatenate([freq, w2v_emb], axis=1).astype(np.float32)  # (N,84+dim)
    if cache:
        os.makedirs(CACHE, exist_ok=True)
        np.savez(cache_f, freq=freq, w2v=w2v_emb, attr=attr)
    log(f"[attr/miRNA] kmer-freq {freq.shape} + w2v {w2v_emb.shape} -> attr {attr.shape}", fh)
    return freq, w2v_emb, attr


# --------------------------------------------------------------------------- #
# channel 1b -- drug attributes: mol2vec substitute (SMILES word2vec)
# --------------------------------------------------------------------------- #
_SMILES_RE = None


def smiles_tokenize(smi):
    """Standard SMILES atom/substructure regex tokenizer (SmilesPE-style)."""
    global _SMILES_RE
    import re
    if _SMILES_RE is None:
        _SMILES_RE = re.compile(
            r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/"
            r"|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])")
    return [t for t in _SMILES_RE.findall(smi) if t]


def drug_mol2vec_substitute(smiles, dim=64, seed=42, cache=True, fh=None):
    """mol2vec substitute: Word2Vec on SMILES token sequences, mean pooled."""
    cache_f = os.path.join(CACHE, f"drug_m2v_dim{dim}_seed{seed}.npz")
    if cache and os.path.isfile(cache_f):
        log(f"[attr/drug] mol2vec-sub substitute cache hit {cache_f}", fh)
        return np.load(cache_f)["attr"]
    corpus = [smiles_tokenize(s) for s in smiles]
    w2v = Word2Vec(sentences=corpus, vector_size=dim, window=5, min_count=1,
                   sg=1, workers=4, epochs=50, seed=seed, negative=5)
    emb = np.stack([
        np.mean([w2v.wv[t] for t in toks], axis=0) for toks in corpus
    ]).astype(np.float32)
    if cache:
        os.makedirs(CACHE, exist_ok=True)
        np.savez(cache_f, attr=emb)
    log(f"[attr/drug] mol2vec substitute {emb.shape} (substitute for unavailable pretrained mol2vec)", fh)
    return emb


# --------------------------------------------------------------------------- #
# channel 1c -- drug GIN graph features (RDKit)
# --------------------------------------------------------------------------- #
def one_of_k(x, allowable):
    if x not in allowable:
        raise Exception(f"{x} not in {allowable}")
    return [x == s for s in allowable]


def one_of_k_unk(x, allowable):
    if x not in allowable:
        x = allowable[-1]
    return [x == s for s in allowable]


ATOM_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                'Pt', 'Hg', 'Pb', 'Unknown']
BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]


def atom_features(atom):
    return np.array(
        one_of_k_unk(atom.GetSymbol(), ATOM_SYMBOLS)
        + one_of_k(atom.GetDegree(), list(range(11)))
        + one_of_k_unk(atom.GetTotalNumHs(), list(range(11)))
        + one_of_k_unk(atom.GetImplicitValence(), list(range(11)))
        + [atom.GetIsAromatic()], dtype=np.float32)


def smile_to_graph(smi):
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    feats = []
    for atom in mol.GetAtoms():
        f = atom_features(atom)
        feats.append(f / (f.sum() + 1e-8))
    x = torch.tensor(np.stack(feats), dtype=torch.float32)
    src, dst, eattr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ba = [float(bond.GetBondType() == bt) for bt in BOND_TYPES] + [float(bond.GetIsConjugated())]
        src += [i, j]
        dst += [j, i]
        eattr += [ba, ba]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(eattr, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def build_drug_graphs(smiles, cache=True, fh=None):
    cache_f = os.path.join(CACHE, "drug_graphs.pt")
    if cache and os.path.isfile(cache_f):
        log(f"[attr/drug] GIN graphs cache hit {cache_f}", fh)
        return torch.load(cache_f)
    graphs = [smile_to_graph(s) for s in smiles]
    if cache:
        os.makedirs(CACHE, exist_ok=True)
        torch.save(graphs, cache_f)
    log(f"[attr/drug] built {len(graphs)} molecular graphs, atom feat dim={graphs[0].x.shape[1]}", fh)
    return graphs


# --------------------------------------------------------------------------- #
# channel 2 -- heterogeneous / similarity network
# --------------------------------------------------------------------------- #
def gip_similarity(A):
    """Gaussian-kernel (GIP) similarity between rows of A.

    gamma = 1 / ((1/n) * sum_i ||A_i||^2) = n / sum_i ||A_i||^2
    sim_ij = exp(-gamma * ||A_i - A_j||^2)
    """
    A = A.astype(np.float64)
    n = A.shape[0]
    sq = (A ** 2).sum(1)
    gamma = n / (sq.sum() + 1e-12)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (A @ A.T)
    d2 = np.clip(d2, 0, None)
    S = np.exp(-gamma * d2).astype(np.float32)
    np.fill_diagonal(S, 0.0)
    return S


def topk_edges(S, k):
    n = S.shape[0]
    k = min(k, n - 1)
    idx = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    return np.stack([rows, idx.reshape(-1)], axis=0)


def build_graph(A, k_mi=10, k_dr=10, cache=True, fh=None):
    """Return edge_index over 1734 nodes (0..1577 mi, 1578..1733 dr)."""
    cache_f = os.path.join(CACHE, f"graph_km{k_mi}_kd{k_dr}.npz")
    if cache and os.path.isfile(cache_f):
        z = np.load(cache_f)
        log(f"[topo] graph cache hit {cache_f}", fh)
        return z["edge_index"]

    n_mi, n_dr = A.shape
    S_mi = gip_similarity(A)          # (n_mi, n_mi)
    S_dr = gip_similarity(A.T)        # (n_dr, n_dr)
    e_mi = topk_edges(S_mi, k_mi)
    e_dr = topk_edges(S_dr, k_dr) + n_mi
    r, c = np.where(A != 0)
    e_md = np.stack([np.concatenate([r, c + n_mi]),
                     np.concatenate([c + n_mi, r])], axis=0)
    # make sim edges undirected
    e_mi = np.concatenate([e_mi, e_mi[::-1]], axis=1)
    e_dr = np.concatenate([e_dr, e_dr[::-1]], axis=1)
    edge_index = np.concatenate([e_mi, e_dr, e_md], axis=1).astype(np.int64)
    if cache:
        os.makedirs(CACHE, exist_ok=True)
        np.savez(cache_f, edge_index=edge_index)
    log(f"[topo] graph: {n_mi} mi + {n_dr} dr = {n_mi+n_dr} nodes, "
        f"{edge_index.shape[1]} directed edges "
        f"(mi-mi {e_mi.shape[1]}, dr-dr {e_dr.shape[1]}, mi-dr {e_md.shape[1]})", fh)
    return edge_index


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class GINDrug(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=64, n_layers=3, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.lin_in = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINConv(mlp))
        self.lin_out = nn.Linear(hidden, out_dim)

    def forward(self, batch):
        x = F.relu(self.lin_in(batch.x))
        for conv in self.convs:
            x = F.relu(conv(x, batch.edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        h = global_add_pool(x, batch.batch)
        return self.lin_out(h)


class MSFFMDI(nn.Module):
    """Attribute + topological multi-source fusion model."""

    def __init__(self, n_mi, n_dr, p_mi, p_dr, gin_dim=64, d_model=64,
                 topo_channels=64, kernels=(3, 5, 7), n_mi_dr_edges=1, dropout=0.5):
        super().__init__()
        atom_dim = 78
        self.n_mi = n_mi
        self.n_dr = n_dr
        self.gin = GINDrug(atom_dim, hidden=64, out_dim=gin_dim, n_layers=3, dropout=dropout)
        self.mi_proj = nn.Linear(p_mi, d_model)
        self.dr_proj = nn.Linear(p_dr + gin_dim, d_model)
        # 2-layer GAT, 4 heads, concat -> d_model
        self.gats = nn.ModuleList([
            GATConv(d_model, d_model // 4, heads=4, concat=True, dropout=dropout),
            GATConv(d_model, d_model // 4, heads=4, concat=True, dropout=dropout),
        ])
        self.kernels = kernels
        self.convs = nn.ModuleList([
            nn.Conv1d(1, topo_channels, kernel_size=k) for k in kernels
        ])
        topo_dim = topo_channels * len(kernels)
        fuse_dim = d_model + topo_dim
        self.fuse_dim = fuse_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * fuse_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def encode_nodes(self, mi_attr, dr_attr, drug_batch, edge_index):
        g_dr = self.gin(drug_batch)                                  # (n_dr, gin_dim)
        a_mi = self.mi_proj(mi_attr)                                 # (n_mi, d)
        a_dr = self.dr_proj(torch.cat([dr_attr, g_dr], dim=1))       # (n_dr, d)
        attrs = torch.cat([a_mi, a_dr], dim=0)                       # (N, d)
        h = attrs
        for gat in self.gats:
            h = F.relu(gat(h, edge_index))
            h = F.dropout(h, 0.5, training=self.training)
        # multi-scale 1D CNN over each node's d-dim feature map
        x = h.unsqueeze(1)                                           # (N, 1, d)
        outs = []
        for conv in self.convs:
            z = F.relu(conv(x))                                      # (N, C, L)
            z = F.adaptive_max_pool1d(z, 1).squeeze(-1)              # (N, C)
            outs.append(z)
        topo = torch.cat(outs, dim=1)                                # (N, 3C)
        fused = torch.cat([attrs, topo], dim=1)                      # (N, fuse)
        return fused

    def forward(self, fused, mi_idx, dr_idx):
        zm = fused.index_select(0, mi_idx)
        zd = fused.index_select(0, dr_idx)
        pair = torch.cat([zm, zd], dim=1)
        return self.mlp(pair).squeeze(-1)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def compute_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "AUC": roc_auc_score(y, p),
        "AUPR": average_precision_score(y, p),
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


# --------------------------------------------------------------------------- #
# training one fold
# --------------------------------------------------------------------------- #
def train_fold(fold_id, A, sequences, smiles, attr_mi, m2v_dr, drug_graphs,
               edge_index, train, test, args, fh, device):
    set_seed(args.seed + fold_id)
    n_mi, n_dr = A.shape
    tr_pairs, tr_y = train
    te_pairs, te_y = test

    # ---- per-fold PCA (fit on training-fold nodes only) -------------------
    tr_mi_nodes = sorted({m for m, _ in tr_pairs})
    tr_dr_nodes = sorted({d for _, d in tr_pairs})
    p_mi = min(args.pca_mi, len(tr_mi_nodes) - 1, attr_mi.shape[1])
    p_dr = min(args.pca_dr, len(tr_dr_nodes) - 1, m2v_dr.shape[1])
    pca_mi = PCA(n_components=p_mi, random_state=args.seed).fit(attr_mi[tr_mi_nodes])
    pca_dr = PCA(n_components=p_dr, random_state=args.seed).fit(m2v_dr[tr_dr_nodes])
    A_mi_p = torch.tensor(pca_mi.transform(attr_mi), dtype=torch.float32, device=device)
    A_dr_p = torch.tensor(pca_dr.transform(m2v_dr), dtype=torch.float32, device=device)
    log(f"[fold {fold_id}] PCA(mi) 148->{p_mi}, evr_sum={pca_mi.explained_variance_ratio_.sum():.3f} | "
        f"PCA(dr) {m2v_dr.shape[1]}->{p_dr}, evr_sum={pca_dr.explained_variance_ratio_.sum():.3f}", fh)

    # ---- model ------------------------------------------------------------
    model = MSFFMDI(n_mi, n_dr, p_mi, p_dr, dropout=args.dropout).to(device)
    drug_batch = Batch.from_data_list(drug_graphs).to(device)
    ei = torch.tensor(edge_index, dtype=torch.long, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    tr_mi = torch.tensor([m for m, _ in tr_pairs], dtype=torch.long, device=device)
    tr_dr = torch.tensor([d for _, d in tr_pairs], dtype=torch.long, device=device)
    tr_yy = torch.tensor(tr_y, dtype=torch.float32, device=device)
    te_mi = torch.tensor([m for m, _ in te_pairs], dtype=torch.long, device=device)
    te_dr = torch.tensor([d for _, d in te_pairs], dtype=torch.long, device=device)

    n = tr_mi.shape[0]
    bs = args.batch_size
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            fused = model.encode_nodes(A_mi_p, A_dr_p, drug_batch, ei)
            out = model(fused, tr_mi[idx], tr_dr[idx])
            loss = loss_fn(out, tr_yy[idx])
            loss.backward()
            opt.step()
            total += loss.item() * idx.numel()
        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            log(f"[fold {fold_id}] epoch {epoch:3d}/{args.epochs}  train_loss={total / n:.5f}", fh)

    # ---- test -------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        fused = model.encode_nodes(A_mi_p, A_dr_p, drug_batch, ei)
        logits = model(fused, te_mi, te_dr)
        p = torch.sigmoid(logits).cpu().numpy()
    m = compute_metrics(te_y, p)
    log(f"[fold {fold_id}] TEST  " + " ".join(f"{k}={v:.4f}" for k, v in m.items()), fh)
    return m


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--pca_mi", type=int, default=64)
    ap.add_argument("--pca_dr", type=int, default=32)
    ap.add_argument("--k_mi", type=int, default=10)
    ap.add_argument("--k_dr", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--out_json", type=str, default=os.path.join(OUT, "results.json"))
    args = ap.parse_args()
    cache = not args.no_cache
    log_every = 5 if args.smoke else args.log_every
    if args.smoke:
        args.epochs, args.folds, log_every = 2, 1, 1

    os.makedirs(CACHE, exist_ok=True)
    fh = open(os.path.join(BASE, "logs", "msffmdi_train.log"), "a")

    header = ("=" * 78 + "\n"
              "MSFFMDI REIMPLEMENTATION (from abstract; no official code; not the\n"
              "paper's numbers). Independent reimplementation for comparison with\n"
              "our DLST-MDA / DLMVF runs on the shared benchmark.\n"
              f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n" + "=" * 78)
    log(header, fh)
    log(f"config: {vars(args)}", fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})", fh)
    log(f"torch {torch.__version__} | torch_geometric {__import__('torch_geometric').__version__}", fh)

    t0 = time.time()
    A, sequences, smiles, mi_names, dr_names = load_data()
    log(f"data: A={A.shape}, positives={int(A.sum())}, "
        f"miRNAs={len(sequences)}, drugs={len(smiles)}", fh)

    freq, w2v, attr_mi = mirna_attributes(sequences, seed=args.seed, cache=cache, fh=fh)
    m2v_dr = drug_mol2vec_substitute(smiles, seed=args.seed, cache=cache, fh=fh)
    drug_graphs = build_drug_graphs(smiles, cache=cache, fh=fh)
    edge_index = build_graph(A, k_mi=args.k_mi, k_dr=args.k_dr, cache=cache, fh=fh)
    folds = build_folds(A)
    log(f"folds: " + " | ".join(
        f"f{i}: train {len(folds[i][0])} (pos {int(folds[i][1].sum())}/neg {int((1-folds[i][1]).sum())}), "
        f"test {len(folds[i][2])} (pos {int(folds[i][3].sum())}/neg {int((1-folds[i][3]).sum())})"
        for i in range(5)), fh)
    log(f"feature/graph prep took {time.time() - t0:.1f}s", fh)

    results = []
    for i in range(args.folds):
        t1 = time.time()
        tr, te = folds[i][:2], folds[i][2:]
        m = train_fold(i, A, sequences, smiles, attr_mi, m2v_dr, drug_graphs,
                       edge_index, tr, te, args, fh, device)
        m["fold"] = i
        m["time_s"] = time.time() - t1
        results.append(m)

    keys = ["AUC", "AUPR", "Accuracy", "Precision", "Recall", "F1", "MCC"]
    log("\n" + "-" * 78, fh)
    log(f"{'Fold':>4} " + " ".join(f"{k:>10}" for k in keys), fh)
    for m in results:
        log(f"{m['fold']:>4} " + " ".join(f"{m[k]:>10.4f}" for k in keys), fh)
    summary = {}
    log("-" * 78, fh)
    for k in keys:
        vals = np.array([m[k] for m in results])
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
                      "min": float(vals.min()), "max": float(vals.max())}
        log(f"MEAN±STD {k:>10}: {vals.mean():.4f} ± "
            f"{(vals.std(ddof=1) if len(vals) > 1 else 0.0):.4f}  (range {vals.min():.4f}-{vals.max():.4f})", fh)
    log("=" * 78, fh)

    payload = {"config": vars(args), "per_fold": results, "summary": summary,
               "note": "Independent reimplementation from the published abstract; "
                       "no official code; full text paywalled; numbers are NOT the "
                       "paper's reported values and are comparable only to our "
                       "DLST-MDA / DLMVF runs on the shared benchmark."}
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"wrote {args.out_json}", fh)
    fh.close()


if __name__ == "__main__":
    main()
