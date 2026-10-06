"""Technique family 2/3: three new algorithm-only members on the accepted bank.

No new columns.  Same accepted shared-anchor fusion, same frozen reference.

  dcn        : feature-crossing network (DCN-v2 style explicit cross layers), BCE.
  lambdarank : MLP trained with a LambdaRank-style objective, pairwise terms weighted
               by |sigmoid(s_pos)-sigmoid(s_neg)| (top-of-ranking emphasis), instead of
               the flat pairwise loss already tested.
  dro        : MLP trained with group-DRO over drug identities (exponentiated-gradient
               worst-group reweighting), a robustness objective never tried here.

Each has a permuted-label twin so the gate can require the real arm > control.

usage: algo2_families.py <task> <surface> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
ALGO = Path('/data1/licheng/incrd-hg/iNcRD-HG/improve/research/algorithmic_next_20260917')
sys.path.insert(0, str(ALGO))
from algo_common import load_bank, save  # noqa: E402

SEED = 20260918
EPOCHS = 50
BATCH = 256
LR = 1e-3
WD = 1e-4


class DCN(nn.Module):
    def __init__(self, d, low=64, n_cross=2):
        super().__init__()
        self.emb = nn.Linear(d, low)
        self.cross_w = nn.ModuleList(nn.Linear(low, low, bias=False) for _ in range(n_cross))
        self.cross_b = nn.ParameterList(nn.Parameter(torch.zeros(low)) for _ in range(n_cross))
        self.deep = nn.Sequential(nn.Linear(low, 128), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(128, 64), nn.ReLU())
        self.head = nn.Linear(low + 64, 1)

    def forward(self, x):
        x0 = self.emb(x)
        h = x0
        for w, b in zip(self.cross_w, self.cross_b):
            h = x0 * (w(h) + b) + h
        z = self.deep(x0)
        return self.head(torch.cat([h, z], -1)).squeeze(1)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def standardise(x_tr, x_q):
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-8
    return (x_tr - mu) / sd, (x_q - mu) / sd


def fit(kind, x_tr, y_tr, groups, x_val):
    xt, xv = standardise(x_tr.astype(np.float32), x_val.astype(np.float32))
    X = torch.tensor(xt, dtype=torch.float32)
    Y = torch.tensor(y_tr.astype(np.float32))
    G = torch.tensor(groups.astype(np.int64))
    torch.manual_seed(SEED)
    model = DCN(x_tr.shape[1]) if kind == 'dcn' else MLP(x_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    n = len(Y)
    g = torch.Generator().manual_seed(SEED)
    bce = nn.functional.binary_cross_entropy_with_logits
    gids = torch.unique(G)
    q = torch.full((len(gids),), 1.0 / len(gids))
    eta = 0.05
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        model.train()
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            s = model(X[idx])
            y = Y[idx]
            if kind == 'lambdarank':
                sp, sn = s[y == 1], s[y == 0]
                if len(sp) == 0 or len(sn) == 0:
                    continue
                w = (torch.sigmoid(sp)[:, None] - torch.sigmoid(sn)[None, :]).abs().detach() + 1e-3
                loss = (nn.functional.softplus(-(sp[:, None] - sn[None, :])) * w).sum() / w.sum()
            elif kind == 'dro':
                gid = G[idx]
                per = bce(s, y, reduction='none')
                gsum = torch.zeros(len(gids))
                gcnt = torch.zeros(len(gids))
                for k, gv in enumerate(gids):
                    m = gid == gv
                    if m.any():
                        gsum[k] = per[m].mean()
                        gcnt[k] = 1.0
                gl = gsum / gcnt.clamp(min=1.0)
                with torch.no_grad():
                    q = q * torch.exp(eta * gl.detach())
                    q = q / q.sum()
                loss = (q * gl).sum()
            else:
                loss = bce(s, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(xv, dtype=torch.float32)).numpy()


def main(task, sname, out_dir):
    _, d = load_bank(task, sname)
    x_tr, y_tr, x_val, pairs_val = d['x_train'], d['y_train'], d['x_val'], d['pairs_val']
    groups = pairs_val[1][:0]  # placeholder unused
    drug_tr = d['pairs_train'][1]
    rng = np.random.default_rng(SEED)
    store = dict(pairs_val=pairs_val)
    jobs = [('dcn', y_tr, drug_tr, 'dcn'),
            ('lambdarank', y_tr, drug_tr, 'lambdarank'),
            ('dro', y_tr, drug_tr, 'dro'),
            ('lambdarank_perm', rng.permutation(y_tr), drug_tr, 'lambdarank')]
    for label, yy, gg, kind in jobs:
        store['logit_%s_member' % label] = fit(kind, x_tr, yy, gg, x_val).astype(np.float32)
        print('fitted', task, sname, label, flush=True)
    save(out_dir, task, sname, store)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
