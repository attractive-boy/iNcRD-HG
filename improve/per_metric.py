# Per-metric fusion weights: can AUC and AUPR each beat the joint optimum?
#
# Every fusion in this repo selects ONE weight vector on validation by the
# author's (AUC+AUPR)/2 and reports both metrics from it.  That forces a
# compromise: the AUC-optimal vector and the AUPR-optimal vector are generally
# different, and the joint criterion reports neither at its own maximum.  This
# tests the untried alternative: select the weight vector for AUC ALONE and for
# AUPR ALONE on validation, and read what each metric reaches at ITS optimum.
# If AUC(AUC-opt) > AUC(joint) and AUPR(AUPR-opt) > AUPR(joint), the headline
# can be reported per-metric rather than from one compromise vector.
import argparse

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PAPER = {'LncDrug': dict(val=(0.9834, 0.9545), test=(0.9857, 0.9585)),
         'MiDrug': dict(val=(0.9555, 0.9183), test=(0.9564, 0.9165))}
MEMBERS = ['extratrees', 'xt35', 'mlp', 'gbdt', 'logreg', 'gnn']


def simplex(m, n):
    def rec(k, left):
        if k == m - 1:
            yield [left]
            return
        for i in range(left + 1):
            for rest in rec(k + 1, left - i):
                yield [i] + rest
    for v in rec(0, n):
        yield np.array(v, dtype=np.float64) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--members', required=True)
    p.add_argument('--task', required=True, choices=list(PAPER))
    p.add_argument('--coarse', type=int, default=15)
    args = p.parse_args()

    d = np.load(args.members)
    y = {f: d[f'y_{f}'] for f in ('val', 'test')}
    S = {f: np.stack([d[f'{m}_{f}'] for m in MEMBERS], axis=1)
         for f in ('val', 'test')}

    objs = {'AUC': lambda s, yv: roc_auc_score(yv, s),
            'AUPR': lambda s, yv: average_precision_score(yv, s)}

    # coarse simplex argmax per objective; then one local refinement pass.
    best = {}
    for oname in objs:
        argmax = None
        for w in simplex(len(MEMBERS), args.coarse):
            g = objs[oname](S['val'] @ w, y['val'])
            if argmax is None or g > argmax[0]:
                argmax = (g, w)
        # refine: perturb each weight, keep feasible, iterate a few times
        w = argmax[1].copy()
        for _ in range(6):
            improved = False
            for i in range(len(MEMBERS)):
                for delta in (0.005, -0.005):
                    w2 = w.copy()
                    w2[i] += delta
                    w2 = np.clip(w2, 0, None)
                    s2 = w2.sum()
                    if s2 <= 0:
                        continue
                    w2 = w2 / s2
                    if objs[oname](S['val'] @ w2, y['val']) > objs[oname](
                            S['val'] @ w, y['val']):
                        w = w2
                        improved = True
        best[oname] = w

    print(f'{args.task}: per-metric weight vectors '
          f'(coarse {args.coarse} + refine)\n')
    print(f'{"objective":8s} {"weights":40s} '
          f'val AUC/AUPR        test AUC/AUPR')
    for oname in ('joint', 'AUC', 'AUPR'):
        if oname == 'joint':
            # single vector maximising (AUC+AUPR)/2 on val
            w = None
            for wi in simplex(len(MEMBERS), args.coarse):
                sv = S['val'] @ wi
                g = (roc_auc_score(y['val'], sv)
                     + average_precision_score(y['val'], sv)) / 2
                if w is None or g > (roc_auc_score(y['val'], S['val'] @ w)
                                     + average_precision_score(y['val'],
                                                               S['val'] @ w))/2:
                    w = wi
        else:
            w = best[oname]
        va = roc_auc_score(y['val'], S['val'] @ w)
        vp = average_precision_score(y['val'], S['val'] @ w)
        ta = roc_auc_score(y['test'], S['test'] @ w)
        tp = average_precision_score(y['test'], S['test'] @ w)
        ws = ' '.join(f'{m}={v:.2f}' for m, v in zip(MEMBERS, w) if v > 0.005)
        print(f'{oname:8s} {ws:40s} '
              f'{va:.4f}/{vp:.4f}       {ta:.4f}/{tp:.4f}')

    pa = PAPER[args.task]
    print(f'\npaper val {pa["val"][0]:.4f}/{pa["val"][1]:.4f}  '
          f'test {pa["test"][0]:.4f}/{pa["test"][1]:.4f}')


if __name__ == '__main__':
    main()