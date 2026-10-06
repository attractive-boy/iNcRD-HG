# -*- coding: utf-8 -*-
"""Provenance completion: rebuild pair -> {pmids, years, ht, nrec, toks} from the full ncRNADrug curated table (DR_Curated.txt,
29.5k records) instead of iNcRD-HG's 10.5k processed records. Keys are PLMF-MDA names (upper) x DrugBank ids; exact name match
first, stem-level (-3p/-5p stripped) fallback. Effect-matched (resistant for MDR, sensitive for MDS). Written to
data/full_<ds>/data/pair_records.pkl so plmf_warm's Prov can load it by pointing prov.HERE at data/full_<ds>."""
import os, re, sys, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921')
from plmf_splits import load_dataset
d = pd.read_csv('/data1/licheng/incrd-hg/data_external/ncrnadrug_20260922/DR_Curated.txt', sep='\t', dtype=str, encoding='latin1', on_bad_lines='skip', quoting=3)
d = d[d['ncRNA_Type'].str.lower().str.contains('mirna', na=False)].copy()
d['name'] = d['ncRNA_Name'].astype(str).str.upper().str.strip(); d['db'] = d['DrugBank_ID'].astype(str).str.strip()
def stem(n): return re.sub(r'-(3P|5P)$', '', n)
import sys as _sys
GENES = '--genes' in _sys.argv
def toks(row):
    out = set()
    if GENES:
        for c in ('ncRNA_Target_Gene', 'Pathway'):
            v = str(row[c]) if pd.notna(row[c]) else ''
            for t in re.split(r'[;,/]', v.strip('"').lower()):
                t = t.strip()
                if t and t not in ('na', '-'): out.add(('gene:' if c == 'ncRNA_Target_Gene' else 'path:') + t)
    for c in ('Phenotype', 'Condition'):
        v = str(row[c]) if pd.notna(row[c]) else ''
        v = v.lower().replace('cell line (', '').replace(')', '')
        for t in re.split(r'[;|,/]', v):
            t = t.strip()
            if t and t != 'na': out.add(t)
    return out
for ds, eff in (('MDR', 'resistant'), ('MDS', 'sensitive')):
    mir, dru, pos, neg = load_dataset(ds)
    names = mir['miRNA_Name'].astype(str).str.upper().values; dbs = dru['DrugBank_ID'].astype(str).values
    sub = d[d['Effect'].str.lower() == eff]
    by_exact = {}; by_stem = {}
    for _, r in sub.iterrows():
        by_exact.setdefault((r['name'], r['db']), []).append(r); by_stem.setdefault((stem(r['name']), r['db']), []).append(r)
    recs = {}; n_exact = n_stem = 0
    for i, j in pos:
        k = (names[i], dbs[j]); rows = by_exact.get(k)
        if rows: n_exact += 1
        else:
            rows = by_stem.get((stem(names[i]), dbs[j]))
            if rows: n_stem += 1
        if not rows: continue
        pm, yrs, tk = [], [], set(); ht = 0
        for r in rows:
            try:
                p = float(r['PMID'])
                if np.isfinite(p) and p > 0: pm.append(int(p))
            except Exception: pass
            try:
                y = float(r['Published_Year'])
                if np.isfinite(y) and 1990 <= y <= 2030: yrs.append(y)
            except Exception: pass
            ht += int(str(r['Throughput']).lower() == 'high'); tk |= toks(r)
        recs[k] = dict(pmids=sorted(set(pm)), years=yrs, ht=ht, nrec=len(rows), toks=tk)
    out = os.path.join(HERE, 'data', f'full{"g" if GENES else ""}_{ds}', 'data'); os.makedirs(out, exist_ok=True)
    pickle.dump(recs, open(os.path.join(out, 'pair_records.pkl'), 'wb'))
    print(f'{ds}: positives {len(pos)}, with records {len(recs)} ({len(recs)/len(pos):.3f}; exact {n_exact}, stem {n_stem}); PMIDs {len(set(p for v in recs.values() for p in v["pmids"]))}; mean PMIDs/pair {np.mean([len(v["pmids"]) for v in recs.values()]):.2f}; mean toks {np.mean([len(v["toks"]) for v in recs.values()]):.1f}')
