# -*- coding: utf-8 -*-
"""Cold-start splits for the DLST-MDA benchmark (Supp. Table 2).

The paper does not release its cold-start split, so we construct a clearly-documented,
reproducible surrogate.  Two settings, mirroring Supp. Table 2:

  * 'mirna': entities (miRNA) are partitioned into 5 disjoint groups; for fold f the test
    set contains ONLY pairs whose miRNA belongs to group f (unseen miRNA), and the train
    set contains only pairs whose miRNA is in the other groups.  Positives = known
    associations; negatives = an equal number of non-associations, sampled within the
    corresponding entity groups (so the test negatives share the unseen miRNAs).
  * 'drug' : identical but partitioning drugs.

Because the whole association matrix is sparse (8720 / 246168 = 3.5%), for some held-out
miRNAs there may be very few positives; we therefore build the cold splits from the full
association matrix (not the 1:1 author split) and cap negatives at the positive count.

Writes: coldstart/mirna/{train,test}{f}.csv and coldstart/drug/{train,test}{f}.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'repos', 'DLST-MDA-main', 'data')
OUT = os.path.join(HERE, 'coldstart')
SEED = 42


def canonical(s):
    return Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True)


def main():
    ass = pd.read_excel(os.path.join(DATA, 'miRNA_drug_matrix.xlsx'), index_col=0)
    rna = pd.read_excel(os.path.join(DATA, 'miRNA_sequences.xlsx'))
    drg = pd.read_excel(os.path.join(DATA, 'drug_id_smiles.xlsx'))
    prots = list(rna['Sequence'])
    drugs = [canonical(s) for s in drg['smiles']]
    A = ass.values
    po_rows, po_cols = np.where(A == 1)
    ne_rows, ne_cols = np.where(A != 1)
    rng = np.random.RandomState(SEED)

    for setting in ('mirna', 'drug'):
        n_ent = A.shape[0] if setting == 'mirna' else A.shape[1]
        groups = np.array_split(rng.permutation(n_ent), 5)
        rdir = os.path.join(OUT, setting)
        os.makedirs(rdir, exist_ok=True)
        for f in range(5):
            held = set(groups[f].tolist())
            if setting == 'mirna':
                te_pos = np.array([i for i, r in enumerate(po_rows) if r in held])
                tr_pos = np.array([i for i, r in enumerate(po_rows) if r not in held])
                te_neg_pool = np.array([i for i, r in enumerate(ne_rows) if r in held])
                tr_neg_pool = np.array([i for i, r in enumerate(ne_rows) if r not in held])
            else:
                te_pos = np.array([i for i, c in enumerate(po_cols) if c in held])
                tr_pos = np.array([i for i, c in enumerate(po_cols) if c not in held])
                te_neg_pool = np.array([i for i, c in enumerate(ne_cols) if c in held])
                tr_neg_pool = np.array([i for i, c in enumerate(ne_cols) if c not in held])
            # negatives: match positive counts (1:1)
            te_neg = rng.choice(te_neg_pool, size=min(len(te_pos), len(te_neg_pool)), replace=False)
            tr_neg = rng.choice(tr_neg_pool, size=min(len(tr_pos), len(tr_neg_pool)), replace=False)
            for split, pos_i, neg_i in [('train', tr_pos, tr_neg), ('test', te_pos, te_neg)]:
                rows = np.concatenate((po_rows[pos_i], ne_rows[neg_i]))
                cols = np.concatenate((po_cols[pos_i], ne_cols[neg_i]))
                with open(os.path.join(rdir, f'{split}{f}.csv'), 'w') as fh:
                    fh.write('compound_iso_smiles,target_sequence,affinity\n')
                    for r_, c_ in zip(rows, cols):
                        fh.write(f'{drugs[c_]},{prots[r_]},{int(A[r_, c_])}\n')
            print(f'coldstart[{setting}] fold {f}: train {len(tr_pos)+len(tr_neg)} '
                  f'(pos {len(tr_pos)}) test {len(te_pos)+len(te_neg)} (pos {len(te_pos)})',
                  flush=True)
    print('done ->', OUT)


if __name__ == '__main__':
    main()
