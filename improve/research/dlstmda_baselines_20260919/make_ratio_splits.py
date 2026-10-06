# -*- coding: utf-8 -*-
"""Construct the positive:negative = 1:5 and 1:10 splits for the DLST-MDA benchmark.

The author repo only ships the 1:1 split (Positive.txt / Negative.txt, 1744 per fold).
Positives (the 8720 known associations) are kept identical to the author split; negatives
are sampled without replacement from the full non-association pool
(1578*156 - 8720 = 237448), split into 5 disjoint folds so that the union of the four
training folds supplies the training negatives -- mirroring the author 1:1 design.

Writes: ratios/1to{r}/{train,test}{fold}.csv  with columns
        compound_iso_smiles,target_sequence,affinity
"""
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'repos', 'DLST-MDA-main', 'data')
OUT = os.path.join(HERE, 'ratios')
SEED = 42


def canonical(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)


def main():
    ass = pd.read_excel(os.path.join(DATA, 'miRNA_drug_matrix.xlsx'), index_col=0)
    rna = pd.read_excel(os.path.join(DATA, 'miRNA_sequences.xlsx'))
    drg = pd.read_excel(os.path.join(DATA, 'drug_id_smiles.xlsx'))
    prots = list(rna['Sequence'])
    drugs = [canonical(s) for s in drg['smiles']]
    A = ass.values
    assert A.shape == (len(prots), len(drugs))
    po_rows, po_cols = np.where(A == 1)
    ne_rows, ne_cols = np.where(A != 1)
    P = json.load(open(os.path.join(DATA, 'Positive.txt')))
    N = json.load(open(os.path.join(DATA, 'Negative.txt')))
    print(f'A={A.shape} pos={len(po_rows)} neg_pool={len(ne_rows)}', flush=True)

    # ---- sanity: rebuild the author 1:1 fold 0 and compare with the shipped CSV ----
    rows = np.concatenate((po_rows[P[0]], ne_rows[N[0]]))
    cols = np.concatenate((po_cols[P[0]], ne_cols[N[0]]))
    shipped = pd.read_csv(os.path.join(DATA, 'processed', 'test0.csv'))
    rebuilt = {(drugs[c], prots[r]) for r, c in zip(rows, cols)}
    ship = {(s, q) for s, q in zip(shipped.compound_iso_smiles, shipped.target_sequence)}
    print(f'  rebuild 1:1 fold0 == shipped: {rebuilt == ship} '
          f'(|rebuild|={len(rebuilt)} |ship|={len(ship)})', flush=True)

    rng = np.random.RandomState(SEED)
    for ratio in (5, 10):
        rdir = os.path.join(OUT, f'1to{ratio}')
        os.makedirs(rdir, exist_ok=True)
        need = ratio * len(po_rows)                 # total negatives
        perm = rng.permutation(len(ne_rows))[:need]
        per_fold = ratio * 1744
        for f in range(5):
            # test negatives = this fold's slice; train negatives = all other slices
            te_neg = perm[f * per_fold:(f + 1) * per_fold]
            tr_neg = np.concatenate([perm[:f * per_fold], perm[(f + 1) * per_fold:]])
            # positives
            tr_pos = np.concatenate([P[j] for j in range(5) if j != f])
            te_pos = np.asarray(P[f])
            tr_rows = np.concatenate((po_rows[tr_pos], ne_rows[tr_neg]))
            tr_cols = np.concatenate((po_cols[tr_pos], ne_cols[tr_neg]))
            te_rows = np.concatenate((po_rows[te_pos], ne_rows[te_neg]))
            te_cols = np.concatenate((po_cols[te_pos], ne_cols[te_neg]))
            for split, rr, cc in [('train', tr_rows, tr_cols), ('test', te_rows, te_cols)]:
                with open(os.path.join(rdir, f'{split}{f}.csv'), 'w') as fh:
                    fh.write('compound_iso_smiles,target_sequence,affinity\n')
                    for r_, c_ in zip(rr, cc):
                        fh.write(f'{drugs[c_]},{prots[r_]},{int(A[r_, c_])}\n')
            print(f'  ratio 1:{ratio} fold {f}: train {len(tr_rows)} (pos {int(A[tr_rows,tr_cols].sum())}) '
                  f'test {len(te_rows)} (pos {int(A[te_rows,te_cols].sum())})', flush=True)
    print('done ->', OUT)


if __name__ == '__main__':
    main()
