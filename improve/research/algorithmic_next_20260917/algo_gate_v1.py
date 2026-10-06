"""Scheme F: pair-adaptive (label-free) shrinkage gate.

The accepted fusion uses one global weight w.  This scheme makes the weight a function
of how much the reference and the member pool disagree on the query pair:

    d   = |logit(ref) - mean_m logit(member_m)|
    w(pair) = w0 * (1 - sigmoid((d - d0)/T))
    score = sigmoid((1 - w(pair)) * logit(ref) + w(pair) * mean_m logit(member_m))

i.e. shrink toward the members when they agree with the reference, and stay close to
the reference when they disagree.  Nothing is trained and no label is used, so the gate
is evaluated on all seven splits with a single setting.  Prints the same gate summary
as the trained schemes.

usage: algo_gate_v1.py
"""
import sys
from pathlib import Path
import numpy as np
from scipy.special import expit

sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG')
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/provenance_next_20260917')
from improve import tabular_foundation as base  # noqa: E402
from provenance_channel_v1 import surfaces, logit  # noqa: E402

CELLS = Path('/data1/licheng/incrd-hg/iNcRD-HG/improve/research/provenance_next_20260917/cells')
MEMBERS = ('hgb', 'et', 'mlp', 'lr')
TASKS = ('MiDrug', 'LncDrug')
GROUP = {'inner': 'inner', 's42': 'resplit', 's2026': 'resplit', 's31415': 'resplit',
         'x7': 'extra', 'x77': 'extra', 'x777': 'extra'}


def rows_for(setting):
    w0, d0, T = setting
    out = []
    for task in TASKS:
        for s in surfaces(task):
            with np.load(CELLS / ('%s_%s_context76.npz' % (task, s['name'])), allow_pickle=False) as z:
                lm = np.mean([z['logit_bank_%s' % m] for m in MEMBERS], axis=0)
            for seed, refpath, key, ykey, pkey in s['refs']:
                with np.load(refpath, allow_pickle=False) as r:
                    ref, y, pairs = r[key], r[ykey], r[pkey]
                lr = logit(ref)
                d = np.abs(lr - lm)
                w = w0 * (1.0 - expit((d - d0) / T))
                v = expit((1 - w) * lr + w * lm)
                m, mr = base.metrics(y, v, pairs), base.metrics(y, ref, pairs)
                out.append(dict(task=task, group=GROUP[s['name']], d_auc=m['auc'] - mr['auc'], d_ap=m['ap'] - mr['ap']))
    return out


def summarize(setting):
    import pandas as pd
    t = pd.DataFrame(rows_for(setting))
    mi, ln = t[t.task == 'MiDrug'], t[t.task == 'LncDrug']
    inner = mi[mi.group == 'inner']
    mi_inner_ok = inner.d_auc.mean() > 0 and inner.d_ap.mean() > 0 and (inner.d_auc > 0).sum() >= 2 and (inner.d_ap > 0).sum() >= 2
    rest = mi[mi.group != 'inner']
    strict = mi_inner_ok and (rest.d_auc > 0).all() and (rest.d_ap > 0).all() and (ln.d_auc >= 0).all() and (ln.d_ap >= 0).all()
    tol = mi_inner_ok and (rest.d_auc > 0).all() and (rest.d_ap > 0).all() and (ln.d_auc >= -0.001).all() and (ln.d_ap >= -0.001).all()
    print('w0=%.2f d0=%.2f T=%.2f strict=%s tol=%s | Mi inner %.5f/%.5f resplit %.5f/%.5f extra %.5f/%.5f | Ln inner %.5f/%.5f resplit %.5f/%.5f extra %.5f/%.5f | Mi_min %.5f/%.5f Ln_min %.5f/%.5f' % (
        setting[0], setting[1], setting[2], strict, tol,
        mi[mi.group == 'inner'].d_auc.mean(), mi[mi.group == 'inner'].d_ap.mean(),
        mi[mi.group == 'resplit'].d_auc.mean(), mi[mi.group == 'resplit'].d_ap.mean(),
        mi[mi.group == 'extra'].d_auc.mean(), mi[mi.group == 'extra'].d_ap.mean(),
        ln[ln.group == 'inner'].d_auc.mean(), ln[ln.group == 'inner'].d_ap.mean(),
        ln[ln.group == 'resplit'].d_auc.mean(), ln[ln.group == 'resplit'].d_ap.mean(),
        ln[ln.group == 'extra'].d_auc.mean(), ln[ln.group == 'extra'].d_ap.mean(),
        mi.d_auc.min(), mi.d_ap.min(), ln.d_auc.min(), ln.d_ap.min()))


if __name__ == '__main__':
    for w0 in (0.2, 0.3, 0.4):
        for d0 in (0.5, 1.0, 2.0):
            for T in (0.25, 0.5):
                summarize((w0, d0, T))
