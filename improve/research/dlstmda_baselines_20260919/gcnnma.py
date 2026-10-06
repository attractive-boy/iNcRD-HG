# -*- coding: utf-8 -*-
"""GCNNMMA / SMiR-GCNN (Niu et al., Frontiers in Genetics 2023) reproduction.

Official code: https://github.com/niuzheyu123/GCNNMMA (SMiR_GCNN.zip).
Faithful batched re-implementation of the official model / pre-processing:
  drug : r-radius (WL) substructure fingerprints -> relational GNN (mean pool)
  miRNA: n-gram word embeddings -> attention-CNN conditioned on the drug vector
  head : 2-class softmax (binary: association / non-association)
Official hyper-parameters (run_training.sh): radius=2, ngram=3, dim=10, layer_gnn=3,
window=11, layer_cnn=3, layer_output=3, lr=1e-3, lr_decay=0.5, decay_interval=10,
weight_decay=1e-6, iteration=100.
"""
import argparse
import csv
import os
import statistics as st
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEV, RESULTS, LOGS, load_fold, seed_it, compute_metrics

RADIUS = 2
NGRAM = 3
DIM = 10
LAYER_GNN = 3
SIDE = 5
WINDOW = 2 * SIDE + 1
LAYER_CNN = 3
LAYER_OUTPUT = 3
LR = 1e-3
LR_DECAY = 0.5
DECAY_INTERVAL = 10
WEIGHT_DECAY = 1e-6
ITERATION = 100
BATCH = 256


# ---------------- official pre-processing (verbatim logic) ---------------------------
def create_atoms(mol, atom_dict):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        atoms[a.GetIdx()] = (atoms[a.GetIdx()], 'aromatic')
    return np.array([atom_dict[a] for a in atoms])


def create_ijbonddict(mol, bond_dict):
    d = defaultdict(list)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        d[i].append((j, bond))
        d[j].append((i, bond))
    return d


def extract_fingerprints(atoms, i_jbond_dict, fingerprint_dict, edge_dict, radius):
    if (len(atoms) == 1) or (radius == 0):
        return np.array([fingerprint_dict[a] for a in atoms])
    nodes = atoms
    i_jedge_dict = i_jbond_dict
    for _ in range(radius):
        fingerprints = []
        for i, j_edge in i_jedge_dict.items():
            neighbors = [(nodes[j], edge) for j, edge in j_edge]
            fingerprints.append(fingerprint_dict[(nodes[i], tuple(sorted(neighbors)))])
        nodes = fingerprints
        _i_jedge_dict = defaultdict(list)
        for i, j_edge in i_jedge_dict.items():
            for j, edge in j_edge:
                both_side = tuple(sorted((nodes[i], nodes[j])))
                edge = edge_dict[(both_side, edge)]
                _i_jedge_dict[i].append((j, edge))
        i_jedge_dict = _i_jedge_dict
    return np.array(fingerprints)


def split_sequence(sequence, ngram, word_dict):
    sequence = '-' + sequence + '='
    return [word_dict[sequence[i:i + ngram]] for i in range(len(sequence) - ngram + 1)]


def build_vocab(pairs):
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    word_dict = defaultdict(lambda: len(word_dict))

    compounds = {}
    words_list = {}
    for smiles, seq in pairs:
        if smiles not in compounds:
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            atoms = create_atoms(mol, atom_dict)
            i_jbond = create_ijbonddict(mol, bond_dict)
            fp = extract_fingerprints(atoms, i_jbond, fingerprint_dict, edge_dict, RADIUS)
            adj = np.array(Chem.GetAdjacencyMatrix(mol), dtype=np.float32)
            compounds[smiles] = (torch.tensor(fp, dtype=torch.long),
                                 torch.tensor(adj, dtype=torch.float32))
        if seq not in words_list:
            words_list[seq] = split_sequence(seq, NGRAM, word_dict)
    return compounds, words_list, len(fingerprint_dict), len(word_dict)


class PairDS(torch.utils.data.Dataset):
    def __init__(self, pairs, labels, comp_index, words_list, lmax):
        self.pairs = pairs
        self.labels = [int(round(float(v))) for v in labels]
        self.comp_index = comp_index
        self.words_list = words_list
        self.lmax = lmax

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        smiles, seq = self.pairs[i]
        words = np.zeros(self.lmax, dtype=np.int64)
        w = self.words_list[seq]
        words[:len(w)] = w
        return self.comp_index[smiles], torch.tensor(words), self.labels[i]


def make_collate(compounds):
    def collate(batch):
        bi, words, ys = zip(*batch)
        nmax = max(compounds[compounds_list[b]][1].size(0) for b in bi)
        B = len(bi)
        fp = torch.zeros(B, nmax, dtype=torch.long)
        adj = torch.zeros(B, nmax, nmax, dtype=torch.float32)
        mask = torch.zeros(B, nmax, dtype=torch.float32)
        for r, b in enumerate(bi):
            f, a = compounds[compounds_list[b]]
            n = a.size(0)
            fp[r, :n] = f
            adj[r, :n, :n] = a
            mask[r, :n] = 1.0
        return fp, adj, mask, torch.stack(words), torch.tensor(ys, dtype=torch.long)
    return collate


class GCNNMMA(nn.Module):
    def __init__(self, n_fingerprint, n_word, dim=DIM, layer_gnn=LAYER_GNN,
                 window=WINDOW, layer_cnn=LAYER_CNN, layer_output=LAYER_OUTPUT):
        super().__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(1, 1, kernel_size=2 * window + 1,
                                              stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2 * dim, 2 * dim) for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2 * dim, 2)

    def gnn(self, fp, adj, mask):
        x = self.embed_fingerprint(fp)              # B,N,dim
        x = x * mask.unsqueeze(-1)
        for i in range(len(self.W_gnn)):
            hs = torch.relu(self.W_gnn[i](x))       # B,N,dim
            x = x + torch.bmm(adj, hs)              # A @ hs
            x = x * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / denom   # mean pool

    def attention_cnn(self, compound, words):
        xs = words.permute(0, 2, 1).unsqueeze(1)    # B,1,dim,L
        for i in range(len(self.W_cnn)):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = xs.squeeze(1).permute(0, 2, 1)         # B,L,dim
        h = torch.relu(self.W_attention(compound))  # B,dim
        hs = torch.relu(self.W_attention(xs))       # B,L,dim
        weights = torch.tanh(torch.bmm(h.unsqueeze(1), hs.permute(0, 2, 1)).squeeze(1))  # B,L
        ys = weights.unsqueeze(-1) * hs
        return ys.mean(dim=1)

    def forward(self, fp, adj, mask, words):
        compound = self.gnn(fp, adj, mask)
        protein = self.attention_cnn(compound, self.embed_word(words))
        cat = torch.cat((compound, protein), 1)
        for j in range(len(self.W_out)):
            cat = torch.relu(self.W_out[j](cat))
        return self.W_interaction(cat)


def run_fold(fd, comp_index, words_list, lmax, n_fp, n_word, seed, device, log,
             epochs=ITERATION):
    tr_pairs, tr_y, te_pairs, te_y = fd
    collate = make_collate(compounds)
    tr_ds = PairDS(tr_pairs, tr_y, comp_index, words_list, lmax)
    te_ds = PairDS(te_pairs, te_y, comp_index, words_list, lmax)
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH, shuffle=True, collate_fn=collate)
    te_loader = torch.utils.data.DataLoader(te_ds, batch_size=512, shuffle=False, collate_fn=collate)
    seed_it(seed)
    model = GCNNMMA(n_fp, n_word).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = nn.CrossEntropyLoss()
    best, last = {'AUC': -1}, None
    y_np = np.array([int(round(float(v))) for v in te_y])
    for ep in range(1, epochs + 1):
        if ep % DECAY_INTERVAL == 0:
            for g in opt.param_groups:
                g['lr'] *= LR_DECAY
        model.train()
        tot = 0.0
        for fp, adj, mask, words, yb in tr_loader:
            fp, adj, mask, words, yb = fp.to(device), adj.to(device), mask.to(device), words.to(device), yb.to(device)
            opt.zero_grad()
            out = model(fp, adj, mask, words)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * yb.size(0)
        model.eval()
        ps = []
        with torch.no_grad():
            for fp, adj, mask, words, yb in te_loader:
                prob = F.softmax(model(fp.to(device), adj.to(device), mask.to(device),
                                       words.to(device)), dim=1)[:, 1].cpu().numpy()
                ps.append(prob)
        m = compute_metrics(y_np, np.concatenate(ps))
        m['epoch'] = ep
        last = m
        if m['AUC'] > best['AUC']:
            best = m
        if ep % 10 == 0 or ep == 1:
            msg = (f"  gcnnma ep{ep:03d} loss{tot/len(tr_y):.4f} AUC{m['AUC']:.4f} "
                   f"AUPR{m['AUPR']:.4f} best{best['AUC']:.4f}")
            print(msg, flush=True)
            log.write(msg + '\n')
            log.flush()
    return best, last


compounds = {}
compounds_list = []


def main():
    global compounds, compounds_list
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--folds', type=str, default='0,1,2,3,4')
    ap.add_argument('--epochs', type=int, default=ITERATION)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    device = DEV
    folds = [int(x) for x in args.folds.split(',')]

    fold_data = {}
    all_pairs = []
    for f in range(5):
        fd = load_fold(f)
        fold_data[f] = fd
        all_pairs += fd[0] + fd[2]
    all_pairs = list(dict.fromkeys(all_pairs))
    print(f'building GCNNMMA vocab over {len(all_pairs)} unique pairs ...', flush=True)
    compounds, words_list, n_fp, n_word = build_vocab(all_pairs)
    compounds_list = list(compounds.keys())
    comp_index = {s: i for i, s in enumerate(compounds_list)}
    lmax = max(len(v) for v in words_list.values())
    print(f'  n_fingerprint={n_fp} n_word={n_word} n_compounds={len(compounds)} '
          f'max_words={lmax}', flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    out_csv = args.out or os.path.join(RESULTS, 'gcnnma_per_fold.csv')
    fields = ['dataset', 'method', 'fold', 'seed', 'auc', 'aupr', 'acc', 'precision',
              'recall', 'specificity', 'f1', 'mcc', 'best_epoch', 'last_auc', 'last_aupr']
    rows = []
    tag = os.environ.get('DLST_TAG', '')
    with open(os.path.join(LOGS, f'gcnnma{tag}_seed{args.seed}.log'), 'w') as log:
        for f in folds:
            t0 = time.time()
            best, last = run_fold(fold_data[f], comp_index, words_list, lmax, n_fp, n_word,
                                  args.seed, device, log, epochs=args.epochs)
            rows.append({'dataset': 'DLST-MDA-1:1', 'method': 'GCNNMMA', 'fold': f,
                         'seed': args.seed, 'auc': best['AUC'], 'aupr': best['AUPR'],
                         'acc': best['ACC'], 'precision': best['Precision'],
                         'recall': best['Recall'], 'specificity': best['Specificity'],
                         'f1': best['F1'], 'mcc': best['MCC'], 'best_epoch': best['epoch'],
                         'last_auc': last['AUC'], 'last_aupr': last['AUPR']})
            msg = (f'== gcnnma fold={f} best_ep={best["epoch"]} best AUC={best["AUC"]:.4f} '
                   f'AUPR={best["AUPR"]:.4f} ACC={best["ACC"]:.4f} ({time.time()-t0:.0f}s)')
            print(msg, flush=True)
            log.write(msg + '\n')
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print('\n=== GCNNMMA ===')
    for k in ['auc', 'aupr', 'acc', 'f1', 'mcc']:
        v = [r[k] for r in rows]
        sd = st.stdev(v) if len(v) > 1 else 0.0
        print(f'  {k}: {st.mean(v):.4f} +/- {sd:.4f}')
    print('saved', out_csv)


if __name__ == '__main__':
    main()
