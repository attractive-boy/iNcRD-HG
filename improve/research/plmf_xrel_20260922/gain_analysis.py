# -*- coding: utf-8 -*-
"""Where does provenance completion land? Positive-side loss by miRNA support degree (seed 42, 5cv), phl vs phlf (vs base);
plus denoised evaluation (documented hidden positives removed from test negatives)."""
import os, sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); WARM = '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/plmf_warm_20260922'
variants = sys.argv[1:] or ['base', 'phl', 'phlf', 'phlf_cluster']
for ds in ('MDR', 'MDS'):
    flags = pd.read_csv(os.path.join(WARM, 'results', f'negflags_{ds}_s42.csv')); hid = set(zip(flags[(flags.in_other == 1) | (flags.in_midrug == 1)].r, flags[(flags.in_other == 1) | (flags.in_midrug == 1)].d))
    print(f'\n=== {ds} (seed 42, 5cv)')
    tab = {}
    for v in variants:
        f = os.path.join(HERE, 'results', f'scores_{ds}_s42_K10_{v}.csv')
        if not os.path.exists(f): print('missing', v); continue
        s = pd.read_csv(f); s['hidden'] = [(r, d) in hid for r, d in zip(s.r, s.d)]
        if 'deg_r' not in s.columns:     # score files from probe_extraneg.py: derive the miRNA support degree from the fold's training positives
            sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921'); from plmf_splits import get_split
            s['deg_r'] = 0
            for fo in s.fold.unique():
                tr, ytr, te, yte = get_split(ds, '5cv', int(fo), 42); dr = np.bincount(tr[ytr == 1][:, 0], minlength=100000)
                s.loc[s.fold == fo, 'deg_r'] = dr[s.loc[s.fold == fo, 'r'].values]
        loss = []; den = []
        for _, g in s.groupby('fold'):
            neg = np.sort(g[g.y == 0].score.values); p = g[g.y == 1]
            l = 1 - np.searchsorted(neg, p.score.values, side='left') / len(neg); loss.append(pd.DataFrame({'loss': l, 'deg_r': p.deg_r.values}))
            gg = g[~((g.y == 0) & g.hidden)]; den.append((roc_auc_score(gg.y, gg.score), average_precision_score(gg.y, gg.score)))
        loss = pd.concat(loss); loss['b'] = pd.cut(loss.deg_r, [-1, 0, 1, 3, 7, 15, 1e9], labels=['0', '1', '2-3', '4-7', '8-15', '16+'])
        tab[v] = loss.groupby('b').loss.mean(); den = np.array(den)
        print(f'{v:14s} denoised AUC/AUPR {den[:,0].mean():.4f} / {den[:,1].mean():.4f}   overall 1-AUC {loss.loss.mean():.4f}')
    t = pd.DataFrame(tab); print('mean positive loss by miRNA support degree:'); print(t.round(4).to_string())
