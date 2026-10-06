"""Scheme S2: study-set Transformer with cross-attention.

Instead of collapsing an entity's curated studies into scalar counts (provenance
channel) this scheme keeps each entity's study set and learns a set-to-set
representation:

  * every curated record is a token = MiniLM(text_ctx 384) ++ [year, throughput,
    effect-resistant, effect-sensitive];
  * the RNA's study set and the drug's study set are each encoded by a permutation-
    invariant Transformer encoder and mean-pooled to h_r / h_d;
  * a cross-attention head lets the RNA pool attend over the drug's study tokens and
    vice versa (c_r / c_d), so the pair score depends on how the two entities'
    *study contexts* align, not on aggregate counts;
  * score = MLP([h_r, h_d, c_r, c_d]).

Cross-fitting follows the bank protocol: for a training row in fold k the study sets
are built from training pairs with fold k removed; query rows use all training pairs.
Control: the whole record-feature matrix is permuted across records (entity<->text map
destroyed, marginals kept).

usage: algo_setx_v1.py <task> <surface> <out_dir>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/provenance_next_20260917')
from algo_common import load_bank, save  # noqa: E402

torch.set_num_threads(1)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SEED = 20260917
CAP = 48
DIM = 64
EPOCHS = 40
BATCH = 128
LR = 1e-3
ROOT = Path('/data1/licheng/incrd-hg/iNcRD-HG/improve/research')
RECORDS = ROOT / 'exposure_next_20260917/records.csv'
TEXT = ROOT / 'provenance_next_20260917/text_title.npy'


def record_features(task, permute):
    rec = pd.read_csv(RECORDS)
    mask = (rec.task == task).values
    rec = rec[mask].reset_index(drop=True)
    emb = np.load(TEXT).astype(np.float32)[mask]
    year = pd.to_numeric(rec.year, errors='coerce').fillna(rec.year.median()).values.astype(np.float32)
    year = (year - year.mean()) / (year.std() + 1e-8)
    ht = rec.ht.fillna(0).values.astype(np.float32)
    eff = rec.Effect.fillna('').astype(str).str.lower().values
    er = (eff == 'resistant').astype(np.float32)
    es = (eff == 'sensitive').astype(np.float32)
    feat = np.c_[emb, year, ht, er, es].astype(np.float32)
    if permute:
        rng = np.random.default_rng(SEED + 3)
        feat = feat[rng.permutation(len(feat))]
    ri = rec.ri.values.astype(np.int64)
    di = rec.di.values.astype(np.int64)
    return feat, ri, di


def build_index(ri, di, support):
    p_r, p_d = {}, {}
    for i in range(len(ri)):
        if (int(ri[i]), int(di[i])) in support:
            p_r.setdefault(int(ri[i]), []).append(i)
            p_d.setdefault(int(di[i]), []).append(i)
    rng = np.random.default_rng(SEED)
    for dct in (p_r, p_d):
        for k, v in dct.items():
            if len(v) > CAP:
                dct[k] = list(rng.choice(v, CAP, replace=False))
    return p_r, p_d


def pad(ids, K, null):
    arr = np.full(K, null, dtype=np.int64)
    m = np.zeros(K, dtype=bool)
    n = min(len(ids), K)
    if n:
        arr[:n] = ids[:n]
        m[:n] = True
    else:
        m[0] = True
    return arr, m


def encode_rows(pairs, p_r, p_d, K, null):
    ir = np.zeros((pairs.shape[1], K), dtype=np.int64)
    mr = np.zeros((pairs.shape[1], K), dtype=bool)
    id_ = np.zeros((pairs.shape[1], K), dtype=np.int64)
    md = np.zeros((pairs.shape[1], K), dtype=bool)
    for j, (r, d) in enumerate(zip(pairs[0], pairs[1])):
        ir[j], mr[j] = pad(p_r.get(int(r), []), K, null)
        id_[j], md[j] = pad(p_d.get(int(d), []), K, null)
    return ir, mr, id_, md


class SetX(nn.Module):
    def __init__(self, in_dim, x_dim):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, DIM), nn.ReLU(), nn.Linear(DIM, DIM))
        layer = nn.TransformerEncoderLayer(DIM, 4, 2 * DIM, dropout=0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, 1)
        self.cross = nn.MultiheadAttention(DIM, 4, dropout=0.1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(4 * DIM + x_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def pool(self, x, mask):
        x = x * mask.unsqueeze(-1)
        return x.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)

    def forward(self, e, ir, mr, id_, md, x):
        xr = self.tr(self.enc(e[ir]), src_key_padding_mask=~mr)
        xd = self.tr(self.enc(e[id_]), src_key_padding_mask=~md)
        hr, hd = self.pool(xr, mr), self.pool(xd, md)
        cr, _ = self.cross(hr.unsqueeze(1), xd, xd, key_padding_mask=~md, need_weights=False)
        cd, _ = self.cross(hd.unsqueeze(1), xr, xr, key_padding_mask=~mr, need_weights=False)
        z = torch.cat([hr, hd, cr.squeeze(1), cd.squeeze(1), x], 1)
        return self.head(z).squeeze(1)


def main(task, sname, out_dir):
    _, d = load_bank(task, sname)
    ptr, y, fold_ids, pq = d['pairs_train'], d['y_train'], d['fold_ids'], d['pairs_val']
    support_full = set(zip(ptr[0].tolist(), ptr[1].tolist()))
    store = dict(pairs_val=pq)
    for label, permute in (('setx', False), ('setx_perm', True)):
        feat, ri, di = record_features(task, permute)
        null = len(feat)
        e = torch.tensor(np.vstack([feat, np.zeros((1, feat.shape[1]), np.float32)])).to(DEVICE)
        ir = np.zeros((len(y), CAP), dtype=np.int64); mr = np.zeros((len(y), CAP), dtype=bool)
        id_ = np.zeros((len(y), CAP), dtype=np.int64); md = np.zeros((len(y), CAP), dtype=bool)
        for k in np.unique(fold_ids):
            q = np.flatnonzero(fold_ids == k)
            sup = support_full - set(zip(ptr[0, q].tolist(), ptr[1, q].tolist()))
            p_r, p_d = build_index(ri, di, sup)
            ar, amr, ad, amd = encode_rows(ptr[:, q], p_r, p_d, CAP, null)
            ir[q], mr[q], id_[q], md[q] = ar, amr, ad, amd
        p_r, p_d = build_index(ri, di, support_full)
        qr, qmr, qd, qmd = encode_rows(pq, p_r, p_d, CAP, null)
        torch.manual_seed(SEED)
        model = SetX(feat.shape[1], d['x_train'].shape[1]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        E = e
        Y = torch.tensor(y.astype(np.float32)).to(DEVICE)
        mu, sd = d['x_train'].mean(0), d['x_train'].std(0) + 1e-8
        XTR = torch.tensor(((d['x_train'] - mu) / sd).astype(np.float32)).to(DEVICE)
        XQ = torch.tensor(((d['x_val'] - mu) / sd).astype(np.float32)).to(DEVICE)
        IR, MR, IDt, MD = (torch.tensor(ir).to(DEVICE), torch.tensor(mr).to(DEVICE), torch.tensor(id_).to(DEVICE), torch.tensor(md).to(DEVICE))
        QR, QMR, QD, QMD = (torch.tensor(qr).to(DEVICE), torch.tensor(qmr).to(DEVICE), torch.tensor(qd).to(DEVICE), torch.tensor(qmd).to(DEVICE))
        g = torch.Generator().manual_seed(SEED)
        n = len(Y)
        for ep in range(EPOCHS):
            perm = torch.randperm(n, generator=g)
            for b in range(0, n, BATCH):
                idx = perm[b:b + BATCH].to(DEVICE)
                s = model(E, IR[idx], MR[idx], IDt[idx], MD[idx], XTR[idx])
                loss = nn.functional.binary_cross_entropy_with_logits(s, Y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            store['logit_%s_member' % label] = model(E, QR, QMR, QD, QMD, XQ).cpu().numpy().astype(np.float32)
        print('fitted', task, sname, label, flush=True)
    save(out_dir, task, sname, store)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
