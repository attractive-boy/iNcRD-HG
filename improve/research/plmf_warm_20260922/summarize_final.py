# -*- coding: utf-8 -*-
"""Summarise results/final/scores_*.csv: standard AUC/AUPR, denoised (documented-positive negatives removed),
sparse-miRNA loss, paired deltas vs base per seed."""
import os, re, glob, sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921')
from plmf_splits import load_dataset
MI = pd.read_csv('/data1/licheng/incrd-hg/incrd-run/2.construct_network/MiDrug_edge.csv')
mi_pairs = set(zip(MI['ncRNA_Name'].str.upper(), MI['DrugBank_ID'].astype(str)))
FLAG = {}
for ds, other in (('MDR', 'MDS'), ('MDS', 'MDR')):
    mir, dru, pos, neg = load_dataset(ds); omir, odru, opos, _ = load_dataset(other)
    names = mir['miRNA_Name'].astype(str).str.upper().values; dbs = dru['DrugBank_ID'].astype(str).values
    onames = omir['miRNA_Name'].astype(str).str.upper().values; odbs = odru['DrugBank_ID'].astype(str).values
    op = set((onames[i], odbs[j]) for i, j in opos)
    FLAG[ds] = (names, dbs, op)

def metrics(df, ds):
    names, dbs, op = FLAG[ds]
    key = list(zip(names[df.r.values], dbs[df.d.values]))
    doc = np.array([(k in op) or (k in mi_pairs) for k in key])
    out = []
    for f, g in df.groupby('fold'):
        d = doc[g.index.values] if False else None
        gi = g.reset_index(drop=True); dd = np.array([(k in op) or (k in mi_pairs) for k in zip(names[gi.r.values], dbs[gi.d.values])])
        auc, ap = roc_auc_score(gi.y, gi.score), average_precision_score(gi.y, gi.score)
        keep = ~((gi.y == 0) & dd)
        auc_d, ap_d = roc_auc_score(gi.y[keep], gi.score[keep]), average_precision_score(gi.y[keep], gi.score[keep])
        negs = np.sort(gi.score[gi.y == 0].values); pos = gi[gi.y == 1]
        above = 1 - np.searchsorted(negs, pos.score.values, side='left') / len(negs)
        lo = pos.deg_r.values <= 1
        out.append((auc, ap, auc_d, ap_d, above[lo].mean(), above[~lo].mean()))
    return np.array(out).mean(0)

rows = []
for p in sorted(glob.glob(os.path.join(HERE, 'results', 'final', 'scores_*.csv'))):
    m = re.match(r'scores_(MDR|MDS)_(single)?s(\d+)_K(\d+)_(.+)\.csv', os.path.basename(p))
    if not m: continue
    ds, single, seed, K, var = m.groups()
    df = pd.read_csv(p)
    if 'deg_r' not in df: continue
    a = metrics(df, ds)
    rows.append(dict(ds=ds, split='single' if single else '5cv', seed=int(seed), K=int(K), variant=var, auc=a[0], aupr=a[1], auc_denoised=a[2], aupr_denoised=a[3], loss_sparse=a[4], loss_dense=a[5]))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(HERE, 'results', 'final', 'metrics_all.csv'), index=False)
pd.set_option('display.width', 250)
order = ['base', 'pmid_only', 'prov', 'mprior', 'final_lean', 'final_noxdeg', 'final', 'final_pub', 'final_shuf']
for (ds, split, K), g in R.groupby(['ds', 'split', 'K']):
    print(f'\n===== {ds} {split} K={K} (seeds: {sorted(g.seed.unique())}) =====')
    base = g[g.variant == 'base'].set_index('seed')
    tab = []
    for v in order:
        gv = g[g.variant == v].set_index('seed')
        if len(gv) == 0: continue
        common = gv.index.intersection(base.index)
        d_auc = (gv.loc[common, 'auc'] - base.loc[common, 'auc']); d_ap = (gv.loc[common, 'aupr'] - base.loc[common, 'aupr'])
        tab.append(dict(variant=v, n=len(gv), AUC=gv.auc.mean(), AUPR=gv.aupr.mean(), dAUC=d_auc.mean(), dAUC_sd=d_auc.std(ddof=0), dAUPR=d_ap.mean(), dAUPR_sd=d_ap.std(ddof=0),
                        wins=f'{int((d_auc > 0).sum())}/{len(common)}', AUC_denoised=gv.auc_denoised.mean(), AUPR_denoised=gv.aupr_denoised.mean(), loss_sparse=gv.loss_sparse.mean(), loss_dense=gv.loss_dense.mean()))
    print(pd.DataFrame(tab).round(4).to_string(index=False))
