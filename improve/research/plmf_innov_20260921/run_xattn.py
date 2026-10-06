# -*- coding: utf-8 -*-
"""Cross-modal co-attention single model on PLMF-MDA (MDR/MDS).

miRNA branch : overlapping 3-mer tokens (step 1) -> embedding + position -> Transformer.
drug  branch : Morgan(r2,2048) on-bit indices     -> token embedding  -> Transformer.
fusion       : bidirectional co-attention (miRNA<->drug) + pooled pair head.

Only the three PLMF-MDA data types are used (sequence / SMILES-derived Morgan / labels).
Same splits as plmf_baselines_20260920 (via mphgnn_plmf_20260921/plmf_splits.py).
Reports test-best (same optimistic protocol as incrd_baseline_plmf) and last epoch.
"""
import argparse, os, sys, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, 'mphgnn_plmf_20260921'))
from plmf_splits import get_split, load_dataset  # noqa

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=['MDR', 'MDS'])
ap.add_argument('--split', default='5cv')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--epochs', type=int, default=200)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--wd', type=float, default=1e-4)
ap.add_argument('--d', type=int, default=128)
ap.add_argument('--heads', type=int, default=4)
ap.add_argument('--layer', type=int, default=2)
ap.add_argument('--ktok', type=int, default=3)
ap.add_argument('--ltok', type=int, default=64)
ap.add_argument('--dropout', type=float, default=0.1)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--tag', default='xattn')
ap.add_argument('--folds', default='')
a = ap.parse_args()
DEV = torch.device(a.device)


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def kmer_ids(seq, k):
    code = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    out = [0]
    for i in range(len(seq) - k + 1):
        c, ok = 0, True
        for ch in seq[i:i + k]:
            b = code.get(ch)
            if b is None:
                ok = False; break
            c = c * 4 + b
        if ok:
            out.append(c + 1)
    return out


def build_tokens(ds):
    g = np.load(os.path.join(BASE, 'mphgnn_plmf_20260921', 'cache', f'graph_{ds}.npz'), allow_pickle=True)
    names = list(g['mirna_names'])
    star = np.load(os.path.join(BASE, 'plmf_strongest_20260920', f'graph_cache_{ds}.npz'), allow_pickle=True)
    morgan = (star['xd'] > 0.5)
    mir, dru, pos, neg = load_dataset(ds)
    name2seq = {str(n): str(s).upper().replace('T', 'U') for n, s in
                zip(mir['miRNA_Name'], mir['miRNA_Sequence'])}
    Lr = 2 + max(len(kmer_ids(name2seq[n], a.ktok)) for n in names)
    Lr = min(Lr, 40)
    R = np.zeros((len(names), Lr), dtype=np.int64)
    for i, n in enumerate(names):
        ids = kmer_ids(name2seq[n], a.ktok)[:Lr]
        R[i, :len(ids)] = ids
    # drug: on-bit indices of Morgan, evenly subsampled to ltok
    D = np.zeros((morgan.shape[0], a.ltok), dtype=np.int64)
    for i in range(morgan.shape[0]):
        bits = np.where(morgan[i])[0]
        bits = bits + 1                      # 0 = pad
        if len(bits) > a.ltok:
            sel = np.linspace(0, len(bits) - 1, a.ltok).round().astype(int)
            bits = bits[sel]
        D[i, :len(bits)] = bits
    return torch.as_tensor(R), torch.as_tensor(D), int(4 ** a.ktok) + 1, morgan.shape[1] + 1


class Branch(nn.Module):
    def __init__(self, vocab, d, layers, heads, dropout, maxlen, ntok):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.pos = nn.Embedding(maxlen, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, tok):
        B, L = tok.shape
        x = self.emb(tok) + self.pos(torch.arange(L, device=tok.device))[None]
        pad = tok.eq(0)
        h = self.enc(x, src_key_padding_mask=pad)
        m = (~pad).float().unsqueeze(-1)
        return h, m


class CoAttn(nn.Module):
    def __init__(self, rvocab, dvocab, Lr, Ld, d, heads, layers, ntok, dropout):
        super().__init__()
        self.mir = Branch(rvocab, d, layers, heads, dropout, Lr, ntok)
        self.drg = Branch(dvocab, d, layers, heads, dropout, Ld, ntok)
        self.m2d = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.d2m = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.Linear(4 * d, 2 * d), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, R, D, r, dd):
        Hr_all, mr = self.mir(R)
        Hd_all, md = self.drg(D)
        Hr = Hr_all[r]; mr = mr[r]          # N,Lr,d
        Hd = Hd_all[dd]; md = md[dd]        # N,Ld,d
        A, _ = self.m2d(Hr, Hd, Hd, key_padding_mask=(md.squeeze(-1) == 0))
        B, _ = self.d2m(Hd, Hr, Hr, key_padding_mask=(mr.squeeze(-1) == 0))
        wr = mr / mr.sum(1, keepdim=True).clamp(min=1)
        wd = md / md.sum(1, keepdim=True).clamp(min=1)
        hr = (A * wr).sum(1); hd = (B * wd).sum(1)
        z = torch.cat([hr, hd, hr * hd, (hr - hd).abs()], -1)
        return self.head(z).view(-1)


def run_fold(R, D, ds, fold, seed):
    tr, ytr, te, yte = get_split(ds, a.split, fold, seed)
    set_seed(seed * 100 + fold)
    model = CoAttn(int(R.max()) + 1, int(D.max()) + 1, R.shape[1], D.shape[1],
                   a.d, a.heads, a.layer, 0, a.dropout).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    Rt, Dt = R.to(DEV), D.to(DEV)
    rtr = torch.as_tensor(tr[:, 0], dtype=torch.long, device=DEV)
    dtr = torch.as_tensor(tr[:, 1], dtype=torch.long, device=DEV)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float, device=DEV)
    rte = torch.as_tensor(te[:, 0], dtype=torch.long, device=DEV)
    dte = torch.as_tensor(te[:, 1], dtype=torch.long, device=DEV)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best = None; last = None; t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); opt.zero_grad()
        pred = model(Rt, Dt, rtr, dtr)
        loss = F.binary_cross_entropy_with_logits(pred, ytr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        if ep % 5 == 0 or ep == a.epochs:
            model.eval()
            with torch.no_grad():
                s = torch.sigmoid(model(Rt, Dt, rte, dte)).cpu().numpy()
            m = dict(epoch=ep, auc=roc_auc_score(yte, s), aupr=average_precision_score(yte, s))
            if best is None or m['auc'] > best['auc']:
                best = dict(m)
            last = dict(m)
            if ep % 25 == 0:
                print(f'    ep{ep} loss {loss.item():.4f} test AUC {m["auc"]:.4f} '
                      f'AUPR {m["aupr"]:.4f} best {best["auc"]:.4f}', flush=True)
    print(f'  fold {fold}: best ep{best["epoch"]} AUC {best["auc"]:.4f} AUPR {best["aupr"]:.4f} | '
          f'last AUC {last["auc"]:.4f} AUPR {last["aupr"]:.4f} ({time.time()-t0:.0f}s)', flush=True)
    return dict(dataset=ds, split=a.split, fold=fold, seed=seed, best_auc=best['auc'],
                best_aupr=best['aupr'], best_epoch=best['epoch'], last_auc=last['auc'], last_aupr=last['aupr'])


def main():
    R, D, rvocab, dvocab = build_tokens(a.dataset)
    print(f'{a.dataset}: mir tokens {tuple(R.shape)} drug tokens {tuple(D.shape)}', flush=True)
    folds = list(range(5)) if a.split == '5cv' else [0]
    if a.folds:
        folds = [int(x) for x in a.folds.split(',')]
    rows = [run_fold(R, D, a.dataset, f, a.seed) for f in folds]
    df = pd.DataFrame(rows)
    out = os.path.join(HERE, 'results', f'xattn_{a.tag}_{a.dataset}_{a.split}_seed{a.seed}.csv')
    df.to_csv(out, index=False)
    print(f'## {a.tag} {a.dataset} {a.split} seed{a.seed}: test-best AUC {df.best_auc.mean():.4f} '
          f'AUPR {df.best_aupr.mean():.4f} | last {df.last_auc.mean():.4f}/{df.last_aupr.mean():.4f}', flush=True)


if __name__ == '__main__':
    main()
