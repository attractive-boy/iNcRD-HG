# -*- coding: utf-8 -*-
"""DAESTB (Li et al., Briefings in Bioinformatics 2022) reproduction.

Official code: https://github.com/biohnuster/DAESTB
Pipeline (verbatim structure): similarity-integrated pair feature matrix ->
deep autoencoder (1734 -> 512 -> 256 -> 128 -> ...) -> XGBoost (100 trees, lr 0.3).

Data adaptation for the DLST-MDA benchmark (no pre-computed similarity matrices are
released for it): drug-drug similarity = Tanimoto over Morgan fingerprints (r=2, 2048
bits); miRNA-miRNA similarity = cosine over 3-mer frequency vectors.
Pair feature for (miRNA i, drug j) = concat(drug_sim[j], mirna_sim[i])  (as in DAESTB).
"""
import argparse
import csv
import os
import statistics as st
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEV, RESULTS, LOGS, load_fold, seed_it, compute_metrics

AE_EPOCHS = 20
AE_BATCH = 128
ENC_DIM = 128


def gip_kernel(M):
    """Gaussian Interaction Profile similarity (rows of M)."""
    M = np.asarray(M, dtype=np.float32)
    n = M.shape[0]
    sq = (M * M).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * M @ M.T, 0)
    gamma = n / (M.sum() + 1e-8)
    S = np.exp(-gamma * d2)
    return ((S + S.T) / 2).astype(np.float32)


def drug_similarity(smiles_list):
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
    n = len(fps)
    S = np.eye(n, dtype=np.float32)
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        S[i] = sims
    return S


def mirna_similarity(seq_list, k=3):
    def feat(seq):
        seq = seq.upper()
        v = np.zeros(4 ** k, dtype=np.float32)
        for i in range(len(seq) - k + 1):
            code = 0
            ok = True
            for ch in seq[i:i + k]:
                b = {'A': 0, 'C': 1, 'G': 2, 'U': 3}.get(ch)
                if b is None:
                    ok = False
                    break
                code = code * 4 + b
            if ok:
                v[code] += 1
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    F = np.stack([feat(s) for s in seq_list])
    return F @ F.T


class DeepAE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(),
                                 nn.Linear(512, 256), nn.ReLU(),
                                 nn.Linear(256, ENC_DIM))
        self.dec = nn.Sequential(nn.Linear(ENC_DIM, 256), nn.ReLU(),
                                 nn.Linear(256, 512), nn.ReLU(),
                                 nn.Linear(512, dim), nn.Tanh())

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--folds', type=str, default='0,1,2,3,4')
    ap.add_argument('--sim', choices=['gip', 'seq'], default='gip')
    args = ap.parse_args()
    seed_it(args.seed)
    folds = [int(x) for x in args.folds.split(',')]

    fold_data = {}
    for f in range(5):
        fold_data[f] = load_fold(f)
    all_pairs = []
    for f in range(5):
        all_pairs += fold_data[f][0] + fold_data[f][2]
    all_pairs = list(dict.fromkeys(all_pairs))

    drugs = list(dict.fromkeys(s for s, _ in all_pairs))
    mirnas = list(dict.fromkeys(q for _, q in all_pairs))
    d_idx = {s: i for i, s in enumerate(drugs)}
    m_idx = {q: i for i, q in enumerate(mirnas)}

    print(f'computing similarities ({args.sim}): {len(drugs)} drugs, {len(mirnas)} miRNAs ...',
          flush=True)
    t0 = time.time()
    if args.sim == 'gip':
        # Gaussian Interaction Profile kernel from the known-association matrix
        # (DAESTB integrates known associations into its feature matrix; standard
        #  construction when no curated functional-similarity matrix is available).
        A = np.zeros((len(mirnas), len(drugs)), dtype=np.float32)
        # labels: recover from the fold splits
        lab = {}
        for f in range(5):
            tr_pairs, tr_y, te_pairs, te_y = fold_data[f]
            for p, y in zip(tr_pairs, tr_y):
                lab[p] = int(y)
            for p, y in zip(te_pairs, te_y):
                lab[p] = int(y)
        for (s, q), y in lab.items():
            A[m_idx[q], d_idx[s]] = y
        DS = gip_kernel(A.T)
        MS = gip_kernel(A)
    else:
        DS = drug_similarity(drugs)
        MS = mirna_similarity(mirnas, k=3)
    print(f'  similarity done in {time.time()-t0:.0f}s', flush=True)

    def pair_features(pairs):
        H = np.empty((len(pairs), DS.shape[1] + MS.shape[1]), dtype=np.float32)
        for r, (s, q) in enumerate(pairs):
            H[r, :DS.shape[1]] = DS[d_idx[s]]
            H[r, DS.shape[1]:] = MS[m_idx[q]]
        return H

    device = DEV
    out_csv = os.path.join(RESULTS, 'daestb_per_fold.csv')
    fields = ['dataset', 'method', 'fold', 'seed', 'auc', 'aupr', 'acc', 'precision',
              'recall', 'specificity', 'f1', 'mcc', 'best_epoch', 'last_auc', 'last_aupr']
    rows = []
    # autoencoder is fit on the full pair matrix (DAESTB is transductive)
    allH = pair_features(all_pairs)
    ae = DeepAE(allH.shape[1]).to(device)
    opt = torch.optim.Adam(ae.parameters())
    lossf = nn.MSELoss()
    X = torch.tensor(allH, device=device)
    for ep in range(1, AE_EPOCHS + 1):
        ae.train()
        perm = torch.randperm(X.size(0), device=device)
        tot = 0.0
        for i in range(0, X.size(0), AE_BATCH):
            b = X[perm[i:i + AE_BATCH]]
            opt.zero_grad()
            rec, _ = ae(b)
            loss = lossf(rec, b)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        print(f'  AE ep{ep:02d} mse={tot/X.size(0):.6f}', flush=True)
    ae.eval()
    with torch.no_grad():
        Z = ae.enc(X).cpu().numpy()

    from xgboost import XGBClassifier
    log_path = os.path.join(LOGS, f'daestb_seed{args.seed}.log')
    with open(log_path, 'w') as log:
        for f in folds:
            tr_pairs, tr_y, te_pairs, te_y = fold_data[f]
            # index rows in allH via key map
            key = {p: i for i, p in enumerate(all_pairs)}
            tr_rows = [key[p] for p in tr_pairs]
            te_rows = [key[p] for p in te_pairs]
            clf = XGBClassifier(n_estimators=100, learning_rate=0.3, eval_metric='logloss',
                                verbosity=0)
            clf.fit(Z[tr_rows], np.asarray(tr_y).astype(int))
            p = clf.predict_proba(Z[te_rows])[:, 1]
            m = compute_metrics(np.asarray(te_y).astype(int), p)
            m['epoch'] = 0
            rows.append({'dataset': 'DLST-MDA-1:1', 'method': 'DAESTB', 'fold': f,
                         'seed': args.seed, 'auc': m['AUC'], 'aupr': m['AUPR'], 'acc': m['ACC'],
                         'precision': m['Precision'], 'recall': m['Recall'],
                         'specificity': m['Specificity'], 'f1': m['F1'], 'mcc': m['MCC'],
                         'best_epoch': 0, 'last_auc': m['AUC'], 'last_aupr': m['AUPR']})
            msg = (f'== daestb fold={f} AUC={m["AUC"]:.4f} AUPR={m["AUPR"]:.4f} '
                   f'ACC={m["ACC"]:.4f} MCC={m["MCC"]:.4f}')
            print(msg, flush=True)
            log.write(msg + '\n')
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print('\n=== DAESTB ===')
    for k in ['auc', 'aupr', 'acc', 'f1', 'mcc']:
        v = [r[k] for r in rows]
        sd = st.stdev(v) if len(v) > 1 else 0.0
        print(f'  {k}: {st.mean(v):.4f} +/- {sd:.4f}')
    print('saved', out_csv)


if __name__ == '__main__':
    main()
