"""Model-seed ensemble (logit average) of saved valsel test scores, paired with the single models."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
from scipy.special import logit
tags = sys.argv[1].split(','); split = sys.argv[2] if len(sys.argv) > 2 else '5cv'; seed = sys.argv[3] if len(sys.argv) > 3 else '42'
for ds in ('MDR', 'MDS'):
    S = [pd.read_csv(f'results/{t}_{ds}_{split}_seed{seed}_scores.csv') for t in tags]
    out = []
    for f in sorted(S[0].fold.unique()):
        xs = [s[s.fold == f].sort_values(['m', 'd']).reset_index(drop=True) for s in S]
        y = xs[0].y.values
        ind = [(AU(y, x.score), AP(y, x.score)) for x in xs]
        L = np.mean([logit(np.clip(x.score.values, 1e-7, 1 - 1e-7)) for x in xs], 0)
        out.append((*np.mean(ind, 0), AU(y, L), AP(y, L)))
    o = np.mean(out, 0)
    print(f'{ds}: single mean {o[0]:.4f}/{o[1]:.4f}  ensemble {o[2]:.4f}/{o[3]:.4f}  gain {o[2]-o[0]:+.4f}/{o[3]-o[1]:+.4f}')
