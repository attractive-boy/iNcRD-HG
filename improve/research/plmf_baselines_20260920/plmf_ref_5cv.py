# -*- coding: utf-8 -*-
"""PLMF-MDA reference on the SAME 5-fold / single splits used for the baselines.

Runs the official PLMF-MDA model (patched local ChemBERTa + RNA-FM) with the
frozen-encoder CLS embeddings cached once per dataset, which is numerically
identical to the official per-batch forward but much faster.  Records best-epoch
and last-epoch test metrics, matching the baseline runner's protocol.

Usage:
  python plmf_ref_5cv.py --dataset MDR --split 5cv --seed 42
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
REPO = os.path.join(BASE, 'new5_repro_20260919', 'repos', 'PLMF-MDA-main')
CODE = os.path.join(REPO, 'code')
SITE = os.path.join(BASE, 'new5_repro_20260919', 'venvs', 'plmf_site')
sys.path.insert(0, CODE)
sys.path.insert(0, SITE)
sys.path.insert(0, HERE)

from data_preprocess import load_drug_data, load_mirna_data, Data_class, collate_fn  # noqa: E402
from layers import DrugMiRNAInteractionModel, RNAFMFeatureExtractor, ChemBERTaFeatureExtractor  # noqa: E402
import plmf_data  # noqa: E402
from common import compute_metrics  # noqa: E402  (dlstmda common, via plmf_data path)

RESULTS = os.path.join(HERE, 'results')
LOGS = os.path.join(HERE, 'logs')
DEV = os.environ.get('PLMF_DEV', 'cuda:0')

# caches ----------------------------------------------------------------
_RNA_CLS = None
_RNA_IDX = None
_SMI_CLS = None
_SMI_IDX = None


def _cache_forward_rna(self, rna_sequence):
    idx = torch.tensor([_RNA_IDX[s] for s in rna_sequence], dtype=torch.long)
    device = next(self.projection.parameters()).device
    cls = _RNA_CLS[idx].to(device)
    return self.dropout(self.projection(cls))


def _cache_forward_smi(self, smiles_list):
    idx = torch.tensor([_SMI_IDX[s] for s in smiles_list], dtype=torch.long)
    device = next(self.projection.parameters()).device
    cls = _SMI_CLS[idx].to(device)
    return self.dropout(self.projection(cls))


def build_caches(drug_features, mirna_features, device):
    global _RNA_CLS, _RNA_IDX, _SMI_CLS, _SMI_IDX
    # unique sequences / smiles, stable order
    seqs = sorted({s for s in mirna_features['sequence'].values()})
    smis = sorted({s for s in drug_features['smiles'].values()})
    _RNA_IDX = {s: i for i, s in enumerate(seqs)}
    _SMI_IDX = {s: i for i, s in enumerate(smis)}

    rna = RNAFMFeatureExtractor(output_dim=128).to(device).eval()
    cls_all = []
    with torch.no_grad():
        for i in range(0, len(seqs), 64):
            chunk = [(f's{i+j}', s) for j, s in enumerate(seqs[i:i + 64])]
            labels, strs, toks = rna.batch_converter(chunk)
            toks = toks.to(device)
            rep = rna.model(toks, repr_layers=[rna.model_layer])['representations'][rna.model_layer]
            cls_all.append(rep[:, 0, :].cpu())
    _RNA_CLS = torch.cat(cls_all, 0)

    smi = ChemBERTaFeatureExtractor(output_dim=128).to(device).eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(smis), 64):
            enc = smi.tokenizer(smis[i:i + 64], padding=True, truncation=True,
                                max_length=512, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            outs.append(smi.model(**enc).last_hidden_state[:, 0, :].cpu())
    _SMI_CLS = torch.cat(outs, 0)
    print(f'  cached RNA-FM {tuple(_RNA_CLS.shape)}  ChemBERTa {tuple(_SMI_CLS.shape)}', flush=True)


def load_idx_fold(dataset, split, fold, seed):
    seqs, smiles, pos, neg = plmf_data.load_dataset(dataset)
    if split == '5cv':
        idx, y = plmf_data._balanced_pairs(pos, neg, seed)
        chunks = np.array_split(np.arange(len(idx)), 5)
        te = chunks[fold]
        tr = np.concatenate([chunks[f] for f in range(5) if f != fold])
    else:
        rng = np.random.RandomState(seed)
        pos = pos[rng.permutation(len(pos))]
        neg_sel = neg[rng.permutation(len(neg))[:len(pos)]]
        te_size = int(0.2 * len(pos))
        tr = np.concatenate([pos[:-te_size], neg_sel[:-te_size]])
        te = np.concatenate([pos[-te_size:], neg_sel[-te_size:]])
        n_tr = len(pos) - te_size
        y_tr = np.concatenate([np.ones(n_tr, dtype=np.float64), np.zeros(n_tr, dtype=np.float64)])
        y_te = np.concatenate([np.ones(te_size, dtype=np.float64), np.zeros(te_size, dtype=np.float64)])
        return (tr, y_tr), (te, y_te)
    return (idx[tr], y[tr]), (idx[te], y[te])


def make_loader(idx, y, drug_features, mirna_features, batch, shuffle):
    triple = np.concatenate([idx, y.reshape(-1, 1)], axis=1)
    ds = Data_class(triple, drug_features, mirna_features)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, collate_fn=collate_fn, drop_last=False)


def evaluate(model, loader, device, criterion):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch['drug_graphs'], batch['drug_smiles'],
                        batch['mirna_encoded'].to(device), batch['mirna_seqs'])
            ys.append(batch['labels'].numpy())
            ps.append(out.cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return compute_metrics(y, p), criterion


def run_fold(args, tr_idx, tr_y, te_idx, te_y, drug_features, mirna_features, log):
    device = args.device
    tr_loader = make_loader(tr_idx, tr_y, drug_features, mirna_features, args.batch, True)
    te_loader = make_loader(te_idx, te_y, drug_features, mirna_features, args.batch, False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = DrugMiRNAInteractionModel(args).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()
    best, last = {'AUC': -1}, None
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for batch in tr_loader:
            labels = batch['labels'].to(device)
            out = model(batch['drug_graphs'], batch['drug_smiles'],
                        batch['mirna_encoded'].to(device), batch['mirna_seqs'])
            loss = criterion(out, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * labels.size(0)
        m, _ = evaluate(model, te_loader, device, criterion)
        m['epoch'] = ep
        last = m
        if m['AUC'] > best['AUC']:
            best = m
        if ep % 10 == 0 or ep == 1:
            msg = (f"  plmf {args.dataset} {args.split} f{args.fold} ep{ep:03d} "
                   f"loss{tot/len(tr_idx):.4f} AUC{m['AUC']:.4f} AUPR{m['AUPR']:.4f} "
                   f"best{best['AUC']:.4f}")
            print(msg, flush=True)
            log.write(msg + '\n')
            log.flush()
    return best, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['MDR', 'MDS'])
    ap.add_argument('--split', default='5cv', choices=['5cv', 'single'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--folds', type=str, default='0,1,2,3,4')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--weight_decay', type=float, default=5e-4)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--device', default=DEV)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    args.no_cuda = False
    args.workers = 0
    args.mirna_feature_dim = 128
    args.drug_feature_dim = 128
    args.drug_input_dim = 69
    args.drug_hidden_dim = 128

    # monkeypatch frozen extractors to use cached CLS embeddings
    RNAFMFeatureExtractor.forward = _cache_forward_rna
    ChemBERTaFeatureExtractor.forward = _cache_forward_smi

    drug_features, _ = load_drug_data(os.path.join(PLMF_DATA_DIR, args.dataset,
                                                   f'drug_ID_{"R" if args.dataset == "MDR" else "S"}.xlsx'))
    mirna_features, _ = load_mirna_data(os.path.join(PLMF_DATA_DIR, args.dataset,
                                                     f'miRNA_ID_{"R" if args.dataset == "MDR" else "S"}.xlsx'))
    build_caches(drug_features, mirna_features, args.device)

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    out_csv = args.out or os.path.join(RESULTS, f'plmf_{args.dataset}_{args.split}_seed{args.seed}.csv')
    fields = ['dataset', 'split', 'method', 'fold', 'seed', 'auc', 'aupr', 'acc', 'precision',
              'recall', 'specificity', 'f1', 'mcc', 'best_epoch', 'last_auc', 'last_aupr']
    rows = []
    with open(os.path.join(LOGS, f'plmf_ref_{args.dataset}_{args.split}_seed{args.seed}.log'), 'w') as log:
        for fold in [int(x) for x in args.folds.split(',')]:
            args.fold = fold
            t0 = time.time()
            (tr_idx, tr_y), (te_idx, te_y) = load_idx_fold(args.dataset, args.split, fold, args.seed)
            best, last = run_fold(args, tr_idx, tr_y, te_idx, te_y, drug_features, mirna_features, log)
            rows.append({'dataset': args.dataset, 'split': args.split, 'method': 'PLMF-MDA',
                         'fold': fold, 'seed': args.seed, 'auc': best['AUC'], 'aupr': best['AUPR'],
                         'acc': best['ACC'], 'precision': best['Precision'], 'recall': best['Recall'],
                         'specificity': best['Specificity'], 'f1': best['F1'], 'mcc': best['MCC'],
                         'best_epoch': best['epoch'], 'last_auc': last['AUC'], 'last_aupr': last['AUPR']})
            msg = (f'== PLMF-MDA {args.dataset} {args.split} fold={fold} best_ep={best["epoch"]} '
                   f'best AUC={best["AUC"]:.4f} AUPR={best["AUPR"]:.4f} '
                   f'| last AUC={last["AUC"]:.4f} AUPR={last["AUPR"]:.4f} ({time.time()-t0:.0f}s)')
            print(msg, flush=True)
            log.write(msg + '\n')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print('saved', out_csv)


PLMF_DATA_DIR = plmf_data.PLMF_DATA

if __name__ == '__main__':
    main()
