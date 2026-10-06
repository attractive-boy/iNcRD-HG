"""Denoised evaluation (same convention as ../plmf_warm_20260922/summarize_final.py): drop test negatives that are
documented positives elsewhere (other-phenotype PLMF matrix or iNcRD-HG's ncRNADrug MiDrug edge list), then AUC/AUPR.
usage: python denoised_eval.py --split 5cv --seeds 42 --tags plain,full"""
import argparse, os, sys, glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'mphgnn_plmf_20260921'))
from plmf_splits import load_dataset
ap = argparse.ArgumentParser(); ap.add_argument('--split', default='5cv'); ap.add_argument('--seeds', default='42'); ap.add_argument('--tags', default='plain,lqo,pair,pair_prov,full'); a = ap.parse_args()
MI = pd.read_csv('/data1/licheng/incrd-hg/incrd-run/2.construct_network/MiDrug_edge.csv')
mi_pairs = set(zip(MI.iloc[:, 0].astype(str).str.upper(), MI.iloc[:, 1].astype(str)))
rows = []
for ds, other in (('MDR', 'MDS'), ('MDS', 'MDR')):
    mir, dru, pos, neg = load_dataset(ds); names = mir['miRNA_Name'].astype(str).str.upper().values; dbs = dru['DrugBank_ID'].astype(str).values
    omir, odru, opos, _ = load_dataset(other); on = omir['miRNA_Name'].astype(str).str.upper().values; od = odru['DrugBank_ID'].astype(str).values
    doc = set((on[i], od[j]) for i, j in opos) | mi_pairs
    for tag in a.tags.split(','):
        for seed in a.seeds.split(','):
            f = os.path.join(HERE, 'results', f'{tag}_{ds}_{a.split}_seed{seed}_scores.csv')
            if not os.path.exists(f): continue
            sc = pd.read_csv(f); sc['doc'] = [(names[m], dbs[d]) in doc for m, d in zip(sc.m, sc.d)]
            for fold, g in sc.groupby('fold'):
                keep = g[(g.y == 1) | (~g.doc)]
                rows.append(dict(dataset=ds, tag=tag, seed=int(seed), fold=fold, auc=roc_auc_score(g.y, g.score), aupr=average_precision_score(g.y, g.score),
                                 auc_dn=roc_auc_score(keep.y, keep.score), aupr_dn=average_precision_score(keep.y, keep.score), n_removed=int((g.doc & (g.y == 0)).sum()), n_neg=int((g.y == 0).sum())))
d = pd.DataFrame(rows); pd.set_option('display.width', 200)
print(d.groupby(['dataset', 'tag']).agg(auc=('auc', 'mean'), aupr=('aupr', 'mean'), auc_dn=('auc_dn', 'mean'), aupr_dn=('aupr_dn', 'mean'), removed=('n_removed', 'mean'), n_neg=('n_neg', 'mean')).reindex([(ds, t) for ds in ('MDR', 'MDS') for t in a.tags.split(',')]).dropna().to_string(float_format=lambda v: f'{v:.4f}'))
