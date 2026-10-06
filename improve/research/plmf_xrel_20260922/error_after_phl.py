# -*- coding: utf-8 -*-
"""Where is the residual loss after PHL (seed 42, 5cv)? Negative side: hidden positives, degree profile of top-scored negatives.
Positive side: lowest-scored positives by miRNA/drug degree, side-info coverage."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); WARM = '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/plmf_warm_20260922'
sys.path.insert(0, WARM); sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921')
from plmf_splits import load_dataset
for ds in ('MDR', 'MDS'):
    s = pd.read_csv(os.path.join(HERE, 'results', f'scores_{ds}_s42_K10_phl.csv'))
    flags = pd.read_csv(os.path.join(WARM, 'results', f'negflags_{ds}_s42.csv')); hid = set(zip(flags[(flags.in_other == 1) | (flags.in_midrug == 1)].r, flags[(flags.in_other == 1) | (flags.in_midrug == 1)].d))
    mir, dru, pos, neg = load_dataset(ds); dd_full = np.bincount(pos[:, 1], minlength=len(dru)); dr_full = np.bincount(pos[:, 0], minlength=len(mir))
    s['hidden'] = [(r, d) in hid for r, d in zip(s.r, s.d)]; s['deg_d'] = dd_full[s.d]; s['deg_r_full'] = dr_full[s.r]
    print(f'\n=== {ds}')
    out = []
    for f, g in s.groupby('fold'):
        pos_s = np.sort(g[g.y == 1].score.values); neg_s = g[g.y == 0]
        # negative loss = fraction of positives ranked below it
        loss_n = 1 - np.searchsorted(pos_s, neg_s.score.values, side='right') / len(pos_s)
        neg_s = neg_s.assign(loss=loss_n)
        tot = loss_n.sum()
        top = neg_s.sort_values('loss', ascending=False)
        for q in (50, 200, 1000):
            t = top.head(q); out.append(dict(fold=f, top=q, loss_share=t.loss.sum() / tot, hidden_rate=t.hidden.mean(), deg_r_med=t.deg_r.median(), deg_d_med=t.deg_d.median()))
    o = pd.DataFrame(out).groupby('top').mean(numeric_only=True).drop(columns='fold'); print('negative side: share of negative loss in top-q negatives, hidden-positive rate, median support deg of miRNA / full deg of drug'); print(o.to_string(float_format=lambda x: f'{x:.3f}'))
    # whole negative side by hidden flag
    ns = []
    for f, g in s.groupby('fold'):
        pos_s = np.sort(g[g.y == 1].score.values); n = g[g.y == 0]; l = 1 - np.searchsorted(pos_s, n.score.values, side='right') / len(pos_s)
        ns.append(pd.DataFrame({'hidden': n.hidden.values, 'loss': l, 'deg_r': n.deg_r.values, 'deg_d': n.deg_d.values}))
    ns = pd.concat(ns); print('negatives: n', len(ns), 'hidden', ns.hidden.sum(), 'hidden loss share', f'{ns[ns.hidden].loss.sum() / ns.loss.sum():.3f}')
    ns['bucket_r'] = pd.cut(ns.deg_r, [-1, 0, 1, 3, 7, 15, 1000], labels=['0', '1', '2-3', '4-7', '8-15', '16+']); ns['bucket_d'] = pd.cut(ns.deg_d, [-1, 10, 30, 60, 100, 200, 10000], labels=['<=10', '11-30', '31-60', '61-100', '101-200', '200+'])
    print('negative loss share by miRNA support degree:'); print((ns.groupby('bucket_r').loss.sum() / ns.loss.sum()).round(3).to_string())
    print('negative loss share by drug full degree:'); print((ns.groupby('bucket_d').loss.sum() / ns.loss.sum()).round(3).to_string())
    print('negative loss share by (miRNA bucket x hidden):'); print((ns.groupby(['bucket_r', 'hidden']).loss.sum() / ns.loss.sum()).round(3).unstack().to_string())
    # positive side
    ps = []
    for f, g in s.groupby('fold'):
        neg_s = np.sort(g[g.y == 0].score.values); p = g[g.y == 1]; l = 1 - np.searchsorted(neg_s, p.score.values, side='left') / len(neg_s)
        ps.append(pd.DataFrame({'loss': l, 'deg_r': p.deg_r.values, 'deg_d': p.deg_d.values}))
    ps = pd.concat(ps); ps['bucket_r'] = pd.cut(ps.deg_r, [-1, 0, 1, 3, 7, 15, 1000], labels=['0', '1', '2-3', '4-7', '8-15', '16+']); ps['bucket_d'] = pd.cut(ps.deg_d, [-1, 10, 30, 60, 100, 200, 10000], labels=['<=10', '11-30', '31-60', '61-100', '101-200', '200+'])
    print('positive loss share by miRNA support degree x drug degree:'); print((ps.groupby(['bucket_r', 'bucket_d']).loss.sum() / ps.loss.sum()).round(3).unstack().to_string())
    print('positive mean loss by miRNA support degree x drug degree:'); print(ps.groupby(['bucket_r', 'bucket_d']).loss.mean().round(3).unstack().to_string())
