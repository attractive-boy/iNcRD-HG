"""Scheme B: two-part propensity-offset model.

Global AUC/AP is dominated by a sampling-propensity / popularity term.  Instead of
feeding degree in as another column and letting one network entangle it with the
biology, this scheme makes the decomposition explicit:

    score(r, d) = beta * prior(r, d) + g_theta(x(r, d))
    prior(r, d) = log1p(deg_r) + log1p(deg_d)      (cross-fitted positive degree)

`beta` is learned end-to-end with the residual network `g_theta`, so the model can
put the popularity mass on the one-dimensional offset and spend the network on the
rest.  Same input bank as every accepted member.

Controls: identical model with the prior permuted across rows, and the same network
without the offset (`offset_none`).

usage: algo_offset_v1.py <task> <surface> <out_dir>
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/provenance_next_20260917')
from algo_common import load_bank, save  # noqa: E402

SEED = 20260917
EPOCHS = 80
BATCH = 512
LR = 1e-3
WD = 1e-4


def crossfit_prior(ptr, ytr, fold_ids, pq):
    n_r = int(max(ptr[0].max(), pq[0].max())) + 1
    n_d = int(max(ptr[1].max(), pq[1].max())) + 1
    out = np.zeros(ptr.shape[1], dtype=np.float32)
    for k in np.unique(fold_ids):
        m = fold_ids != k
        pos = ptr[:, m][:, ytr[m] == 1]
        cr = np.bincount(pos[0], minlength=n_r).astype(np.float32)
        cd = np.bincount(pos[1], minlength=n_d).astype(np.float32)
        q = np.flatnonzero(fold_ids == k)
        out[q] = np.log1p(cr[ptr[0, q]]) + np.log1p(cd[ptr[1, q]])
    pos = ptr[:, ytr == 1]
    cr = np.bincount(pos[0], minlength=n_r).astype(np.float32)
    cd = np.bincount(pos[1], minlength=n_d).astype(np.float32)
    outq = np.log1p(cr[pq[0]]) + np.log1p(cd[pq[1]])
    return out, outq.astype(np.float32)


def make_model(d):
    torch.manual_seed(SEED)
    net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
    beta = nn.Parameter(torch.tensor(1.0))
    return net, beta


def fit(x_tr, y_tr, prior_tr, x_val, prior_val, use_offset):
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-8
    pm, ps = prior_tr.mean(), prior_tr.std() + 1e-8
    X = torch.tensor((x_tr - mu) / sd, dtype=torch.float32)
    Xv = torch.tensor((x_val - mu) / sd, dtype=torch.float32)
    P = torch.tensor((prior_tr - pm) / ps, dtype=torch.float32)
    Pv = torch.tensor((prior_val - pm) / ps, dtype=torch.float32)
    Y = torch.tensor(y_tr.astype(np.float32))
    net, beta = make_model(x_tr.shape[1])
    opt = torch.optim.Adam(list(net.parameters()) + [beta], lr=LR, weight_decay=WD)
    n = len(Y)
    g = torch.Generator().manual_seed(SEED)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            s = net(X[idx]).squeeze(1)
            if use_offset:
                s = s + beta * P[idx]
            loss = nn.functional.binary_cross_entropy_with_logits(s, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        s = net(Xv).squeeze(1)
        if use_offset:
            s = s + beta * Pv
    return s.numpy()


def main(task, sname, out_dir):
    _, d = load_bank(task, sname)
    x_tr, y_tr, x_val, pq, ptr = d['x_train'], d['y_train'], d['x_val'], d['pairs_val'], d['pairs_train']
    prior_tr, prior_val = crossfit_prior(ptr, y_tr, d['fold_ids'], pq)
    rng = np.random.default_rng(SEED)
    store = dict(pairs_val=pq, prior_val=prior_val)
    combos = {'offset': (prior_tr, True), 'offset_perm': (rng.permutation(prior_tr), True),
              'offset_none': (prior_tr, False)}
    for label, (pt, use) in combos.items():
        store['logit_%s_member' % label] = fit(x_tr, y_tr, pt, x_val, prior_val, use).astype(np.float32)
        print('fitted', task, sname, label, flush=True)
    save(out_dir, task, sname, store)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
