"""Scheme C: cross-fitted stacking with a pairwise-ranking meta-learner.

The accepted fusion is the *uniform mean* of four member logits, fused with the
reference at a fixed scalar weight.  This scheme replaces the uniform mean with a
learned meta-learner over the same four members:

  * the four accepted members (HGB/ExtraTrees/MLP/LogReg) are fit out-of-fold on the
    training rows (5 folds), giving honest OOF logits;
  * a small meta-network maps the four OOF logits to a score and is trained with the
    same cross-drug pairwise logistic loss as Scheme A;
  * at query time the four members are fit on the full training fold, scored, and the
    meta-network is applied.

Controls: the uniform mean of the identical OOF-fit members (`stackmean`), and the
meta-network trained on permuted labels (`stack_perm`).

usage: algo_stack_v1.py <task> <surface> <out_dir>
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

sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG')
from improve.context_anchor_verify import learner  # noqa: E402

MEMBERS = ('hgb', 'et', 'mlp', 'lr')
SEED = 20260917


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def oof_members(x, y, fold_ids):
    oof = np.zeros((len(y), len(MEMBERS)), dtype=np.float32)
    for k in np.unique(fold_ids):
        tr = fold_ids != k
        q = np.flatnonzero(fold_ids == k)
        for j, name in enumerate(MEMBERS):
            m = learner(name)
            m.fit(x[tr], y[tr])
            oof[q, j] = logit(m.predict_proba(x[q])[:, 1])
    return oof


def full_members(x_tr, y_tr, x_val):
    out = np.zeros((len(x_val), len(MEMBERS)), dtype=np.float32)
    for j, name in enumerate(MEMBERS):
        m = learner(name)
        m.fit(x_tr, y_tr)
        out[:, j] = logit(m.predict_proba(x_val)[:, 1])
    return out


def fit_meta(oof, y, drug):
    torch.manual_seed(SEED)
    net = nn.Sequential(nn.Linear(oof.shape[1], 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=1e-4)
    X = torch.tensor(oof, dtype=torch.float32)
    Y = torch.tensor(y.astype(np.float32))
    D = torch.tensor(drug.astype(np.int64))
    g = torch.Generator().manual_seed(SEED)
    n = len(Y)
    for ep in range(200):
        perm = torch.randperm(n, generator=g)
        for b in range(0, n, 512):
            idx = perm[b:b + 512]
            s = net(X[idx]).squeeze(1)
            yy, dd = Y[idx], D[idx]
            sp, sn = s[yy == 1], s[yy == 0]
            dp, dn = dd[yy == 1], dd[yy == 0]
            if len(sp) == 0 or len(sn) == 0:
                continue
            M = sp[:, None] - sn[None, :]
            valid = (dp[:, None] != dn[None, :]).float()
            loss = (nn.functional.softplus(-M) * valid).sum() / valid.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    return net


def apply_meta(net, x):
    with torch.no_grad():
        return net(torch.tensor(x, dtype=torch.float32)).squeeze(1).numpy()


def main(task, sname, out_dir):
    _, d = load_bank(task, sname)
    x_tr, y_tr, x_val, pq = d['x_train'], d['y_train'], d['x_val'], d['pairs_val']
    fold_ids, drug = d['fold_ids'], d['pairs_train'][1]
    oof = np.nan_to_num(oof_members(x_tr, y_tr, fold_ids), nan=0.0, posinf=0.0, neginf=0.0)
    val = np.nan_to_num(full_members(x_tr, y_tr, x_val), nan=0.0, posinf=0.0, neginf=0.0)
    rng = np.random.default_rng(SEED)
    store = dict(pairs_val=pq, oof_train=oof)
    store['logit_stackmean_member'] = val.mean(1).astype(np.float32)
    store['logit_stack_member'] = apply_meta(fit_meta(oof, y_tr, drug), val).astype(np.float32)
    store['logit_stack_perm_member'] = apply_meta(fit_meta(oof, rng.permutation(y_tr), drug), val).astype(np.float32)
    print('fitted', task, sname, 'stack/stackmean/stack_perm', flush=True)
    save(out_dir, task, sname, store)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
