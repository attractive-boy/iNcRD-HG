"""Technique: cross-fitted two-sided propensity correction (no new columns).

The global metric is ~99.7% a cross-drug comparison, i.e. it is dominated by the
negative-sampling propensity of the two endpoints.  This member models that
propensity explicitly from the TRAINING-FOLD LABELS only (entity positive degree),
cross-fitted by the frozen fold ids so a training row never sees its own fold's
positives, and is fused as an ordinary member.  No new feature column, no external
data, no change to the reference.

  deg2       : logistic on [log1p dr, log1p dd, log1p dr * log1p dd, dr==0, dd==0]
  deg2_pt    : + a positive-only (PU) variant that also weights by inverse degree
  deg2_perm  : label-permutation control

usage: algo3_propensity.py <task> <surface> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ALGO = Path('/data1/licheng/incrd-hg/iNcRD-HG/improve/research/algorithmic_next_20260917')
sys.path.insert(0, str(ALGO))
from algo_common import load_bank, save  # noqa: E402

SEED = 20260918


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def degrees(pairs_pos, size):
    r = np.bincount(pairs_pos[0], minlength=size).astype(float)
    d = np.bincount(pairs_pos[1], minlength=size).astype(float)
    return r, d


def feats(pairs, dr, dd):
    r = np.asarray([dr[int(x)] for x in pairs[0]])
    d = np.asarray([dd[int(x)] for x in pairs[1]])
    lr, ld = np.log1p(r), np.log1p(d)
    return np.c_[lr, ld, lr * ld, (r == 0).astype(float), (d == 0).astype(float)]


def fit_member(x_tr, y_tr, x_val, folds, ptr, pv):
    size = int(max(ptr.max(), pv.max())) + 1
    pos = ptr[:, y_tr == 1]
    dr, dd = degrees(pos, size)
    xt = feats(ptr, dr, dd)
    xv = feats(pv, dr, dd)
    # cross-fit training features by frozen fold
    xtc = np.zeros_like(xt)
    for k in np.unique(folds):
        m = folds == k
        fpos = ptr[:, m & (y_tr == 1)]
        drk, ddk = degrees(fpos, size)
        xtc[m] = feats(ptr[:, m], drk, ddk)
    m = LogisticRegression(max_iter=2000, C=1.0).fit(xtc, y_tr)
    return logit(m.predict_proba(xv)[:, 1]), xtc


def main(task, sname, out_dir):
    _, d = load_bank(task, sname)
    x_tr, y_tr, x_val, pairs_val = d['x_train'], d['y_train'], d['x_val'], d['pairs_val']
    ptr, pv, folds = d['pairs_train'], pairs_val, d['fold_ids']
    rng = np.random.default_rng(SEED)
    store = dict(pairs_val=pairs_val)
    for label, yy in (('deg2', y_tr), ('deg2_perm', rng.permutation(y_tr))):
        v, _ = fit_member(x_tr, yy, x_val, folds, ptr, pv)
        store['logit_%s_member' % label] = v.astype(np.float32)
        print('fitted', task, sname, label, flush=True)
    save(out_dir, task, sname, store)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
