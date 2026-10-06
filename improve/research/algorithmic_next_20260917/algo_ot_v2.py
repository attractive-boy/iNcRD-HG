"""Scheme S3: differentiable optimal-transport (Sinkhorn) head over study sets.

Each entity is a set of study tokens (same construction as the set-Transformer scheme).
Instead of pooling + cross-attention, the pair is scored by solving an entropic
optimal-transport problem between the RNA's study distribution and the drug's study
distribution *inside the network* (log-domain Sinkhorn, differentiable):

    C_ij = 1 - cos(xr_i, xd_j)
    P    = argmin_{P in U(a,b)} <P, C> - eps H(P)
    z    = [ <P, C>, max_{ij: valid} (P_ij), h_r, h_d, alignment stats, bank ]
    score = MLP(z)

i.e. the model learns to score a pair by how cheaply its two study profiles can be
transported onto each other.  Trained end-to-end with BCE.  Cross-fitting and the
permuted-record control follow the other schemes.

usage: algo_ot_v1.py <task> <surface> <out_dir>
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/provenance_next_20260917')
from algo_common import load_bank, save  # noqa: E402
from algo_setx_v1 import record_features, build_index, encode_rows, CAP  # noqa: E402

PROV = Path('/data1/licheng/incrd-hg/iNcRD-HG/improve/research/provenance_next_20260917/cells')

torch.set_num_threads(1)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SEED = 20260917
DIM = 64
EPS = 0.1
ITERS = 20
EPOCHS = 40
BATCH = 128
LR = 1e-3


def sinkhorn(cost, mask_r, mask_d, eps=EPS, iters=ITERS):
    """Log-domain entropic OT with uniform marginals; padded positions masked out."""
    B, K1, K2 = cost.shape
    neg_inf = torch.finfo(cost.dtype).min
    valid = mask_r[:, :, None] & mask_d[:, None, :]
    logK = torch.where(valid, -cost / eps, torch.full_like(cost, neg_inf))
    log_a = torch.log(mask_r / mask_r.sum(1, keepdim=True).clamp(min=1))
    log_b = torch.log(mask_d / mask_d.sum(1, keepdim=True).clamp(min=1))
    u, v = log_a.clone(), log_b.clone()
    for _ in range(iters):
        u = log_a - torch.logsumexp(logK + v[:, None, :], dim=2)
        v = log_b - torch.logsumexp(logK + u[:, :, None], dim=1)
    logP = logK + u[:, :, None] + v[:, None, :]
    P = torch.exp(torch.where(valid, logP, torch.full_like(logP, neg_inf)))
    ot = (P * cost).sum((1, 2))
    return ot, P


class OTNet(nn.Module):
    def __init__(self, in_dim, x_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, DIM), nn.ReLU(), nn.Linear(DIM, DIM))
        self.head = nn.Sequential(nn.Linear(2 * DIM + 5 + x_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
        self.logeps = nn.Parameter(torch.tensor(0.0))

    def forward(self, e, ir, mr, id_, md, x):
        xr = torch.nn.functional.normalize(self.proj(e[ir]), dim=-1)
        xd = torch.nn.functional.normalize(self.proj(e[id_]), dim=-1)
        cos = xr @ xd.transpose(1, 2)
        cost = (1.0 - cos).clamp(min=0.0)
        valid = (mr[:, :, None] & md[:, None, :]).float()
        cost = cost * valid + (1.0 - valid) * 3.0
        eps = torch.exp(self.logeps).clamp(0.02, 1.0)
        ot, P = sinkhorn(cost, mr, md, eps=eps)
        pmax = P.amax((1, 2))
        mass = P.sum((1, 2)).clamp(min=1e-9)
        hr = (xr * mr.unsqueeze(-1)).sum(1) / mr.sum(1, keepdim=True).clamp(min=1)
        hd = (xd * md.unsqueeze(-1)).sum(1) / md.sum(1, keepdim=True).clamp(min=1)
        stats = torch.stack([ot, pmax, mass, eps.expand_as(ot), (ot / mass)], 1)
        z = torch.cat([hr, hd, stats, x], 1)
        return self.head(z).squeeze(1)


def main(task, sname, out_dir):
    _, d = load_bank(task, sname)
    ptr, y, fold_ids, pq = d['pairs_train'], d['y_train'], d['fold_ids'], d['pairs_val']
    support_full = set(zip(ptr[0].tolist(), ptr[1].tolist()))
    store = dict(pairs_val=pq)
    for label, permute in (('ot', False), ('ot_perm', True)):
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
        with np.load(PROV / ('%s_%s_context76.npz' % (task, sname)), allow_pickle=False) as zz:
            ftr, fvl = zz['feat_train_real'], zz['feat_val_real']
        xtr = np.c_[d['x_train'], ftr]; xvl = np.c_[d['x_val'], fvl]
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-8
        XTR = torch.tensor(((xtr - mu) / sd).astype(np.float32)).to(DEVICE)
        XQ = torch.tensor(((xvl - mu) / sd).astype(np.float32)).to(DEVICE)
        torch.manual_seed(SEED)
        model = OTNet(feat.shape[1], d['x_train'].shape[1] + 14).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        Y = torch.tensor(y.astype(np.float32)).to(DEVICE)
        T = lambda a: torch.tensor(a).to(DEVICE)
        IR, MR, IDt, MD = T(ir), T(mr), T(id_), T(md)
        QR, QMR, QD, QMD = T(qr), T(qmr), T(qd), T(qmd)
        g = torch.Generator().manual_seed(SEED)
        n = len(Y)
        for ep in range(EPOCHS):
            perm = torch.randperm(n, generator=g)
            for b in range(0, n, BATCH):
                idx = perm[b:b + BATCH].to(DEVICE)
                s = model(e, IR[idx], MR[idx], IDt[idx], MD[idx], XTR[idx])
                loss = nn.functional.binary_cross_entropy_with_logits(s, Y[idx])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
        model.eval()
        with torch.no_grad():
            store['logit_%s_member' % label] = model(e, QR, QMR, QD, QMD, XQ).cpu().numpy().astype(np.float32)
        print('fitted', task, sname, label, flush=True)
    save(out_dir, task, sname, store)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
