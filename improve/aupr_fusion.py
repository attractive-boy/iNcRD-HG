# Does selecting the fusion weight for AUPR alone buy AUPR?
#
# The final predictor in RESULTS.md selects fusion weights on validation by the
# author's own (AUC+AUPR)/2 criterion, so a single weight vector reports both
# metrics.  AUPR is the metric with the real residual (LncDrug test +0.023,
# ~3x AUC's), and its residual is concentrated at the TOP of the ranking, which
# AUC barely sees.  This tests the obvious untried lever: select the weight for
# AUPR ALONE on validation, and see whether it buys test AUPR over the joint
# criterion.  Both normalisations (rank and raw probability) are compared,
# because AUPR is scale-sensitive at the head while AUC is not.
import argparse

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PAPER = {'LncDrug': dict(val=(0.9834, 0.9545), test=(0.9857, 0.9585)),
         'MiDrug': dict(val=(0.9555, 0.9183), test=(0.9564, 0.9165))}
MEMBERS = ['gnn', 'extratrees', 'logreg', 'gbdt']
HIST = {'LncDrug': ('incrd-orig/9.improve/alt_lnc.npz', 'gnn_lnc.npz'),
        'MiDrug': ('incrd-orig/9.improve/alt_mi.npz', 'gnn_mi.npz')}


def rk(v):
    return v.argsort().argsort().astype(np.float64) / max(len(v) - 1, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True, choices=list(PAPER))
    p.add_argument('--grid', type=int, default=21)
    args = p.parse_args()

    alt, gnnf = HIST[args.task]
    a = np.load(alt)
    y = {f: a[f'y_{f}'] for f in ('val', 'test')}

    S = {m: {f: a[f'{m}_{f}'] for f in ('val', 'test')} for m in MEMBERS}

    # members in the published four-scorer fusion are already in `a`
    norm = {'rank': lambda v: rk(v), 'raw': lambda v: v}

    def metric_fn(metric):
        return (lambda yt, s: roc_auc_score(yt, s)) if metric == 'auc' \
            else (lambda yt, s: average_precision_score(yt, s))

    grid = np.linspace(0, 1, args.grid)
    print(f'{args.task}: weight selection rule x normalisation '
          f'(S_val / S_test, AUC / AUPR)\n')
    for nname, nf in norm.items():
        Sn = {m: {f: nf(S[m][f]) for f in ('val', 'test')} for m in MEMBERS}
        rows = []
        for rule in ('joint', 'aupr', 'auc'):
            best = (None, None)
            for w1 in grid:
                for w2 in grid:
                    for w3 in grid:
                        w4 = 1 - w1 - w2 - w3
                        if w4 < -1e-9:
                            continue
                        ws = (w1, w2, w3, w4)
                        sv = sum(ws[i] * Sn[m]['val'] for i, m in enumerate(MEMBERS))
                        if rule == 'joint':
                            g = (roc_auc_score(y['val'], sv)
                                 + average_precision_score(y['val'], sv)) / 2
                        elif rule == 'aupr':
                            g = average_precision_score(y['val'], sv)
                        else:
                            g = roc_auc_score(y['val'], sv)
                        if best[0] is None or g > best[0]:
                            best = (g, ws)
            g, ws = best
            s = {f: sum(ws[i] * Sn[m][f] for i, m in enumerate(MEMBERS))
                 for f in ('val', 'test')}
            va = roc_auc_score(y['val'], s['val'])
            vp = average_precision_score(y['val'], s['val'])
            ta = roc_auc_score(y['test'], s['test'])
            tp = average_precision_score(y['test'], s['test'])
            rows.append((rule, ws, va, vp, ta, tp))
        print(f'  [{nname}]')
        for rule, ws, va, vp, ta, tp in rows:
            print(f'    {rule:6s} w={" ".join(f"{w:.2f}" for w in ws):24s} '
                  f'val {va:.4f}/{vp:.4f}  test {ta:.4f}/{tp:.4f}')
    pa, pp = PAPER[args.task]['val']
    pa_t, pp_t = PAPER[args.task]['test']
    print(f'\n  paper val {pa:.4f}/{pp:.4f}   test {pa_t:.4f}/{pp_t:.4f}')


if __name__ == '__main__':
    main()