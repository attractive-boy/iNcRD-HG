"""Technique 1 (model class): feature-token Transformer member on the accepted bank.

No new columns.  The accepted member pool (HGB/ExtraTrees/MLP/LogReg) treats the 76
bank columns as a flat vector.  This member instead tokenises each column, adds a
CLS token and runs a small Transformer encoder (feature-wise self-attention), i.e.
a different inductive bias over exactly the same input.  It is a drop-in replacement
member in the accepted shared-anchor fusion.

Control: identical architecture and optimiser, labels permuted; must not pass.

usage: algo1_fttransformer.py <task> <surface> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)

HERE = Path(__file__).resolve().parent
ALGO = Path('/data1/licheng/incrd-hg/iNcRD-HG/improve/research/algorithmic_next_20260917')
sys.path.insert(0, str(ALGO))
from algo_common import load_bank, save  # noqa: E402

SEED = 20260918
EPOCHS = 60
BATCH = 256
LR = 1e-3
WD = 1e-4
D_MODEL = 64
HEADS = 4
LAYERS = 2
DROPOUT = 0.2


class FTTransformer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.feat_w = nn.Parameter(torch.randn(d, D_MODEL) * 0.02)
        self.feat_b = nn.Parameter(torch.zeros(d, D_MODEL))
        self.cls = nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, HEADS, D_MODEL * 2, DROPOUT, batch_first=True, activation='gelu'
        )
        self.enc = nn.TransformerEncoder(layer, LAYERS)
        self.head = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, 1))

    def forward(self, x):
        tok = x.unsqueeze(-1) * self.feat_w.unsqueeze(0) + self.feat_b.unsqueeze(0)
        h = torch.cat([self.cls.expand(x.size(0), -1, -1), tok], dim=1)
        h = self.enc(h)
        return self.head(h[:, 0]).squeeze(1)


def standardise(x_tr, x_q):
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-8
    return (x_tr - mu) / sd, (x_q - mu) / sd


def fit(x_tr, y_tr, x_val):
    xt, xv = standardise(x_tr.astype(np.float32), x_val.astype(np.float32))
    X = torch.tensor(xt, dtype=torch.float32)
    Y = torch.tensor(y_tr.astype(np.float32))
    torch.manual_seed(SEED)
    model = FTTransformer(x_tr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    n = len(Y)
    g = torch.Generator().manual_seed(SEED)
    lossf = nn.functional.binary_cross_entropy_with_logits
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        model.train()
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            s = model(X[idx])
            loss = lossf(s, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(xv, dtype=torch.float32)).numpy()


def main(task, sname, out_dir):
    _, d = load_bank(task, sname)
    x_tr, y_tr, x_val, pairs_val = d['x_train'], d['y_train'], d['x_val'], d['pairs_val']
    rng = np.random.default_rng(SEED)
    store = dict(pairs_val=pairs_val)
    for label, yy in (('fttrans', y_tr), ('fttrans_perm', rng.permutation(y_tr))):
        store['logit_%s_member' % label] = fit(x_tr, yy, x_val).astype(np.float32)
        print('fitted', task, sname, label, flush=True)
    save(out_dir, task, sname, store)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
