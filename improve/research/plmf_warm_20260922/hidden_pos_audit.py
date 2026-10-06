# -*- coding: utf-8 -*-
"""Audit: how many PLMF-MDA sampled negatives are documented positives elsewhere?
Sources: (1) the other phenotype matrix (MDS for MDR and vice versa), (2) iNcRD-HG's
ncRNADrug-derived MiDrug edge list (names + DrugBank IDs), (3) union."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921')
from plmf_splits import load_dataset, _balanced_pairs

MI = pd.read_csv('/data1/licheng/incrd-hg/incrd-run/2.construct_network/MiDrug_edge.csv')
mi_pairs = set(zip(MI['ncRNA_Name'].str.upper(), MI['DrugBank_ID'].astype(str)))
print('iNcRD-HG MiDrug edges:', len(MI), 'unique name/DB pairs:', len(mi_pairs))

def load(ds):
    mir, dru, pos, neg = load_dataset(ds)
    names = mir['miRNA_Name'].astype(str).str.upper().values
    dbs = dru['DrugBank_ID'].astype(str).values
    return names, dbs, pos, neg

out = {}
for ds, other in (('MDR', 'MDS'), ('MDS', 'MDR')):
    names, dbs, pos, neg = load(ds)
    onames, odbs, opos, oneg = load(other)
    other_pairs = set((onames[i], odbs[j]) for i, j in opos)
    pos_pairs = set((names[i], dbs[j]) for i, j in pos)
    print(f'\n=== {ds}: {len(names)} miRNA x {len(dbs)} drug, pos {len(pos)}, neg pool {len(neg)}')
    print(f'  shared miRNA with {other}: {len(set(names)&set(onames))}, shared drugs: {len(set(dbs)&set(odbs))}')
    print(f'  {ds} positives that are also {other} positives: {len(pos_pairs & other_pairs)} ({len(pos_pairs & other_pairs)/len(pos):.3f})')
    print(f'  {ds} positives in iNcRD-HG MiDrug: {len(pos_pairs & mi_pairs)} ({len(pos_pairs & mi_pairs)/len(pos):.3f})')
    # negative pool contamination
    neg_names = [(names[i], dbs[j]) for i, j in neg]
    in_other = np.array([p in other_pairs for p in neg_names])
    in_mi = np.array([p in mi_pairs for p in neg_names])
    print(f'  neg pool: in {other} positives {in_other.sum()} ({in_other.mean():.4f}); in MiDrug {in_mi.sum()} ({in_mi.mean():.4f}); union {(in_other|in_mi).sum()} ({(in_other|in_mi).mean():.4f})')
    # the actual sampled negatives (seed 42 protocol)
    idx, y = _balanced_pairs(pos, neg, 42)
    sel = idx[y == 0]
    sel_names = [(names[i], dbs[j]) for i, j in sel]
    s_other = np.array([p in other_pairs for p in sel_names]); s_mi = np.array([p in mi_pairs for p in sel_names])
    print(f'  sampled negatives (seed42, n={len(sel)}): in {other} pos {s_other.sum()} ({s_other.mean():.4f}); in MiDrug {s_mi.sum()} ({s_mi.mean():.4f}); union {(s_other|s_mi).sum()} ({(s_other|s_mi).mean():.4f})')
    # both-entity-covered rate: negatives whose miRNA and drug both appear in the other source
    cov_other = np.array([(n in set(onames)) and (d in set(odbs)) for n, d in sel_names])
    mi_names = set(MI['ncRNA_Name'].str.upper()); mi_dbs = set(MI['DrugBank_ID'].astype(str))
    cov_mi = np.array([(n in mi_names) and (d in mi_dbs) for n, d in sel_names])
    print(f'  sampled negatives with both entities covered by {other}: {cov_other.mean():.3f}; by MiDrug: {cov_mi.mean():.3f}')
    flag = pd.DataFrame({'r': sel[:, 0], 'd': sel[:, 1], 'in_other': s_other.astype(int), 'in_midrug': s_mi.astype(int)})
    flag.to_csv(f'/data1/licheng/incrd-hg/iNcRD-HG/improve/research/plmf_warm_20260922/results/negflags_{ds}_s42.csv', index=False)
