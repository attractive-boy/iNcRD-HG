"""Two further algorithm-level schemes on the provenance-augmented bank.

Both take the same accepted input `[context76 | 14 provenance columns]` and the same
two-layer MLP body; they differ from `pv_rank`/`pv_bce` in the objective and in the
model structure:

  nce    : listwise cross-drug InfoNCE.  For each positive i,
           -log softmax_i( {s_i} + {s_j : drug_j != drug_i} ) over the batch.
           This is the listwise counterpart of the pairwise loss (Scheme A).
  offset : two-part structure.  score = beta * prior(r,d) + MLP(x), with
           prior = log1p(deg_r)+log1p(deg_d) cross-fitted from training positives and
           beta learned end-to-end (Scheme B, now applied on the augmented bank).

Controls: permuted-label versions of both.

usage: algo_prov_v2.py <task> <surface> <out_dir>
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
from algo_offset_v1 import crossfit_prior  # noqa: E402
from algo_pairwise_v1 import make_model, standardise  # noqa: E402

torch.set_num_threads(1)
SEED = 20260917
EPOCHS = 60
BATCH = 512
LR = 1e-3
WD = 1e-4
PROV = Path('/data1/licheng/incrd-hg/iNcRD-HG/improve/research/provenance_next_20260917/cells')


def train(x_tr, y_tr, drug, x_val, prior_tr, prior_val, mode):
    xt, xv = standardise(x_tr.astype(np.float32), x_val.astype(np.float32))
    X = torch.tensor(xt)
    Y = torch.tensor(y_tr.astype(np.float32))
    D = torch.tensor(drug.astype(np.int64))
    use_offset = mode.startswith('offset')
    if use_offset:
        pm, ps = prior_tr.mean(), prior_tr.std() + 1e-8
        P = torch.tensor(((prior_tr - pm) / ps).astype(np.float32))
        Pv = torch.tensor(((prior_val - pm) / ps).astype(np.float32))
    net = make_model(x_tr.shape[1])
    beta = nn.Parameter(torch.tensor(1.0))
    params = list(net.parameters()) + ([beta] if use_offset else [])
    opt = torch.optim.Adam(params, lr=LR, weight_decay=WD)
    n = len(Y)
    g = torch.Generator().manual_seed(SEED)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            s = net(X[idx]).squeeze(1)
            if use_offset:
                s = s + beta * P[idx]
            yy, dd = Y[idx], D[idx]
            if mode.startswith('nce'):
                sp, sn = s[yy == 1], s[yy == 0]
                dp, dn = dd[yy == 1], dd[yy == 0]
                if len(sp) == 0 or len(sn) == 0:
                    continue
                M = sp[:, None] - sn[None, :]
                valid = (dp[:, None] != dn[None, :])
                M = M.masked_fill(~valid, float('-inf'))
                loss = -torch.logsumexp(torch.cat([torch.zeros(len(sp), 1), M], 1), 1).mean()
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(s, yy)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        s = net(torch.tensor(xv)).squeeze(1)
        if use_offset:
            s = s + beta * Pv
    return s.numpy()


def main(task, sname, out_dir):
    _, d = load_bank(task, sname)
    with np.load(PROV / ('%s_%s_context76.npz' % (task, sname)), allow_pickle=False) as z:
        ptr, pv = z['feat_train_real'], z['feat_val_real']
    x_tr = np.c_[d['x_train'], ptr].astype(np.float32)
    x_val = np.c_[d['x_val'], pv].astype(np.float32)
    y = d['y_train']
    drug = d['pairs_train'][1]
    prior_tr, prior_val = crossfit_prior(d['pairs_train'], y, d['fold_ids'], d['pairs_val'])
    rng = np.random.default_rng(SEED)
    store = dict(pairs_val=d['pairs_val'])
    jobs = {'nce': (y, 'nce'), 'nce_perm': (rng.permutation(y), 'nce'),
            'offset2': (y, 'offset'), 'offset2_perm': (rng.permutation(y), 'offset')}
    for label, (yy, mode) in jobs.items():
        store['logit_%s_member' % label] = train(x_tr, yy, drug, x_val, prior_tr, prior_val, mode).astype(np.float32)
        print('fitted', task, sname, label, flush=True)
    save(out_dir, task, sname, store)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
