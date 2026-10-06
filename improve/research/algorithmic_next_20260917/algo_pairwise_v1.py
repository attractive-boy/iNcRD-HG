"""Scheme A: cross-drug pairwise ranking objective.

The metric is global AUC/AP over all positive x negative test pairs, and 99.7% of
those pairs do NOT share a drug.  The accepted members (HGB/ExtraTrees/MLP/LogReg)
are trained with pointwise log-loss, which does not see that structure.  This scheme
replaces the objective with a sampled-softmax / pairwise logistic loss computed only
over cross-drug pairs, keeping the architecture and the input features identical:

    L = mean_{i positive} mean_{j negative, drug_j != drug_i} softplus(-(s_i - s_j))

Controls: the identical MLP trained with BCE (objective-only comparison) and the
pairwise model trained on permuted labels.

usage: algo_pairwise_v1.py <task> <surface> <out_dir>
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from algo_common import load_bank, save  # noqa: E402

SEED = 20260917
EPOCHS = 60
BATCH = 256
LR = 1e-3
WD = 1e-4


def make_model(d):
    torch.manual_seed(SEED)
    return nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))


def standardise(x_tr, x_q):
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-8
    return (x_tr - mu) / sd, (x_q - mu) / sd


def fit(x_tr, y_tr, d_tr, x_val, objective):
    xt, xv = standardise(x_tr.astype(np.float32), x_val.astype(np.float32))
    X = torch.tensor(xt, dtype=torch.float32)
    Y = torch.tensor(y_tr.astype(np.float32))
    D = torch.tensor(d_tr.astype(np.int64))
    model = make_model(x_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    n = len(Y)
    g = torch.Generator().manual_seed(SEED)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            s = model(X[idx]).squeeze(1)
            y, d = Y[idx], D[idx]
            if objective == 'bce':
                loss = nn.functional.binary_cross_entropy_with_logits(s, y)
            else:
                sp, sn = s[y == 1], s[y == 0]
                dp, dn = d[y == 1], d[y == 0]
                if len(sp) == 0 or len(sn) == 0:
                    continue
                M = sp[:, None] - sn[None, :]
                valid = (dp[:, None] != dn[None, :]).float()
                loss = (nn.functional.softplus(-M) * valid).sum() / valid.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(xv, dtype=torch.float32)).squeeze(1).numpy()


def main(task, sname, out_dir):
    s, d = load_bank(task, sname)
    x_tr, y_tr = d['x_train'], d['y_train']
    x_val, pairs_val = d['x_val'], d['pairs_val']
    drug = d['pairs_train'][1]
    rng = np.random.default_rng(SEED)
    store = dict(pairs_val=pairs_val)
    combos = {'rank': (y_tr, 'pair'), 'rank_bce': (y_tr, 'bce'),
              'rank_perm': (rng.permutation(y_tr), 'pair')}
    for label, (yy, obj) in combos.items():
        v = fit(x_tr, yy, drug, x_val, obj)
        store['logit_%s_member' % label] = v.astype(np.float32)
        print('fitted', task, sname, label, flush=True)
    save(out_dir, task, sname, store)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
