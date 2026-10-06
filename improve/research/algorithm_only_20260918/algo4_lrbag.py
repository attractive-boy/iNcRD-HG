"""Technique: bagged LambdaRank member (new objective + variance reduction), no new columns.

The single LambdaRank member is positive on the MiDrug mean of every split group and
its label-permuted control is negative, but two of seven individual splits fall by
~1e-4 (noise).  This member averages K LambdaRank MLPs fitted with different
initialisations; still algorithm-only (objective + ensemble of one member type).

  lrbag      : mean logit over K seeds, LambdaRank objective
  lrbag_perm : same, labels permuted (control)

usage: algo4_lrbag.py <task> <surface> <out_dir>
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

SEEDS = (20260918, 20260919, 20260920, 20260921, 20260922)
EPOCHS = 50
BATCH = 256
LR = 1e-3
WD = 1e-4


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def fit(seed, x_tr, y_tr, x_val):
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-8
    xt, xv = (x_tr - mu) / sd, (x_val - mu) / sd
    X = torch.tensor(xt.astype(np.float32))
    Y = torch.tensor(y_tr.astype(np.float32))
    torch.manual_seed(seed)
    model = MLP(x_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    n = len(Y)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        model.train()
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            s = model(X[idx])
            y = Y[idx]
            sp, sn = s[y == 1], s[y == 0]
            if len(sp) == 0 or len(sn) == 0:
                continue
            w = (torch.sigmoid(sp)[:, None] - torch.sigmoid(sn)[None, :]).abs().detach() + 1e-3
            loss = (nn.functional.softplus(-(sp[:, None] - sn[None, :])) * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(xv.astype(np.float32))).numpy()


def main(task, sname, out_dir):
    _, d = load_bank(task, sname)
    x_tr, y_tr, x_val, pairs_val = d['x_train'], d['y_train'], d['x_val'], d['pairs_val']
    rng = np.random.default_rng(SEEDS[0])
    store = dict(pairs_val=pairs_val)
    for label, yy in (('lrbag', y_tr), ('lrbag_perm', rng.permutation(y_tr))):
        preds = [fit(s, x_tr, yy, x_val) for s in SEEDS]
        store['logit_%s_member' % label] = np.mean(preds, axis=0).astype(np.float32)
        print('fitted', task, sname, label, flush=True)
    save(out_dir, task, sname, store)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
