# Blend the GNN ensemble with the GBDT, weight selected on validation.
#
# The two models share no inductive bias: one recovers collaborative structure
# through message passing, the other reads it as explicit numbers and splits on
# it.  If their errors are decorrelated the blend beats both, and the rank
# correlation between their scores says up front how much room there is.
#
# The weight is a single scalar chosen on the validation fold -- the same kind
# of validation selection the paper's own Table 2 rests on -- so S_val is
# optimistic in the usual way and S_test, which the weight never sees, is the
# honest column.  Both are printed.
import argparse

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

SCORER = {'auc': roc_auc_score, 'aupr': average_precision_score}
PAPER = {'LncDrug': dict(val=(0.9834, 0.9545), test=(0.9857, 0.9585)),
         'MiDrug': dict(val=(0.9555, 0.9183), test=(0.9564, 0.9165))}


def rk(v):
    return v.argsort().argsort().astype(np.float64) / max(len(v) - 1, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gnn', required=True)
    p.add_argument('--gbdt', required=True)
    p.add_argument('--task', required=True, choices=list(PAPER))
    p.add_argument('--combiner', default='uniform_top8cfg',
                   help='which GNN combiner to blend (default: the one with '
                        'no fitting on any reported fold)')
    p.add_argument('--norm', default='r', choices=['r', 'z'])
    args = p.parse_args()

    g = np.load(args.gnn)
    b = np.load(args.gbdt)
    y = {f: g[f'y_{f}'] for f in ('val', 'test')}
    assert np.array_equal(y['val'], b['y_val']), 'val label order differs'
    assert np.array_equal(y['test'], b['y_test']), 'test label order differs'

    pa = PAPER[args.task]
    grid = np.linspace(0, 1, 51)
    print(f'{args.task}  GNN[{args.combiner},{args.norm}] x GBDT\n')
    for metric in ('auc', 'aupr'):
        S = {f: rk(g[f'{metric}_{args.norm}_{args.combiner}_{f}'])
             for f in ('val', 'test')}
        B = {f: rk(b[f]) for f in ('val', 'test')}
        rho = spearmanr(S['val'], B['val']).correlation
        sc = SCORER[metric]
        vals = [sc(y['val'], (1 - a) * S['val'] + a * B['val']) for a in grid]
        a_star = grid[int(np.argmax(vals))]
        blend = {f: (1 - a_star) * S[f] + a_star * B[f] for f in ('val', 'test')}

        i = 0 if metric == 'auc' else 1
        print(f'-- {metric.upper()} --   spearman(GNN, GBDT) on val = {rho:.3f}'
              f'   alpha* = {a_star:.2f}')
        for lab, s in (('GNN alone   ', S), ('GBDT alone  ', B),
                       ('blend       ', blend)):
            print(f'   {lab} val {sc(y["val"], s["val"]):.4f} '
                  f'({sc(y["val"], s["val"]) - pa["val"][i]:+.4f})   '
                  f'test {sc(y["test"], s["test"]):.4f} '
                  f'({sc(y["test"], s["test"]) - pa["test"][i]:+.4f})')
        print()


if __name__ == '__main__':
    main()
