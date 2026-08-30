# What is the released protocol's global AUC actually made of?
#
# Global AUC is P(score_pos > score_neg) over every positive x negative pair in
# the test fold.  Split that pair set by whether the two edges share a drug:
#
#   same-drug pairs  -- the biology.  Can the model tell which ncRNA is
#                       associated with THIS drug?
#   cross-drug pairs -- can it tell that a positive on a popular drug outranks
#                       a negative on a rare one?  This is the drug prior.
#
# The ratio between the two is a property of the split, not of any model, and
# it bounds how much any genuine improvement can possibly be worth.
import argparse

import torch

TASKS = [('LncDrug', 'lncRNA'), ('MiDrug', 'miRNA')]


def decompose(run_dir, task, rna):
    d = torch.load(f'{run_dir}/3.heter_data/{task}_test_data.pth',
                   map_location='cpu')
    e = d[rna, task, 'drug']
    y = e.edge_label.numpy()
    drug = e.edge_label_index[1].numpy()
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    same = sum(int((y[drug == g] == 1).sum()) * int((y[drug == g] == 0).sum())
               for g in set(drug.tolist()))
    total = n_pos * n_neg
    return n_pos, n_neg, total, same


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    args = p.parse_args()
    for task, rna in TASKS:
        n_pos, n_neg, total, same = decompose(args.run_dir, task, rna)
        frac = same / total
        print(f'{task}: pos={n_pos} neg={n_neg}')
        print(f'  ranking pairs decided by the metric : {total}')
        print(f'  same-drug (biology)                 : {same} ({100*frac:.2f}%)')
        print(f'  cross-drug (drug prior)             : {total-same} '
              f'({100*(1-frac):.2f}%)')
        # A model already at ~0.65 within-drug that became a perfect
        # within-drug ranker would move global AUC by frac * (1 - 0.65).
        print(f'  max global AUC gain from a PERFECT within-drug ranker, '
              f'starting at 0.65: {frac * 0.35:.4f}\n')


if __name__ == '__main__':
    main()
