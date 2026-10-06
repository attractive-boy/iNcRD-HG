# -*- coding: utf-8 -*-
"""How much label signal the test-query graph carries for MPHGNN's global branch.
The official model builds the global GCN edge_index from the (miRNA, drug) pairs of
the batch being scored, so when the test set is scored in batches the batch's own
degree structure is visible. Heuristic score = deg_miRNA(batch) * deg_drug(batch)."""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plmf_splits import get_split
rows = []
for ds in ('MDR', 'MDS'):
    for split, seeds in (('single', [0, 1, 2, 3, 4]), ('5cv', [42])):
        for sd in seeds:
            for f in (range(5) if split == '5cv' else [0]):
                tr, ytr, te, yte = get_split(ds, split, f, sd)
                nm = max(te[:, 0].max(), tr[:, 0].max()) + 1; nd = max(te[:, 1].max(), tr[:, 1].max()) + 1
                for cs in (1, 8, 32, 128, 256, 1024, 10 ** 9):
                    s = np.zeros(len(te))
                    for st in range(0, len(te), cs):
                        c = te[st:st + cs]; a = np.bincount(c[:, 0], minlength=nm); b = np.bincount(c[:, 1], minlength=nd)
                        s[st:st + cs] = a[c[:, 0]] * b[c[:, 1]]
                    rows.append(dict(dataset=ds, split=split, seed=sd, fold=f, batch=('full' if cs == 10 ** 9 else cs), auc=roc_auc_score(yte, s)))
                dm = np.bincount(tr[ytr == 1, 0], minlength=nm); dd = np.bincount(tr[ytr == 1, 1], minlength=nd)
                rows.append(dict(dataset=ds, split=split, seed=sd, fold=f, batch='train-graph', auc=roc_auc_score(yte, dm[te[:, 0]] * dd[te[:, 1]])))
d = pd.DataFrame(rows)
t = d.groupby(['dataset', 'split', 'batch']).auc.mean().unstack('batch')
t = t[[1, 8, 32, 128, 256, 1024, 'full', 'train-graph']]
print(t.round(3).to_string())
t.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', '_query_graph_leak.csv'))
