"""Record-level curation metadata for MiDrug/LncDrug pairs (label-free w.r.t. any split).
Joins ncRNADrug DR_Curated.txt records to node indices and to the Cellosaurus-normalised
disease in incrd-run/13.context/{MiDir,LncDir}_context.csv (record index == row order of
the task's records in DR_Curated after the same filtering used by direction_labels.py)."""
import re, numpy as np, pandas as pd
ROOT = '/data1/licheng/incrd-hg/incrd-run/'
raw = pd.read_csv(ROOT + '10.circ/raw/DR_Curated.txt', sep='\t', encoding='latin-1', low_memory=False)
raw['cid'] = pd.to_numeric(raw.CID, errors='coerce')
drugs = pd.read_csv(ROOT + '2.construct_network/Drug.csv'); d2i = {int(c): i for i, c in enumerate(drugs.CID)}
out = []
for task, rtype, rfile, ctx in [('MiDrug', 'miRNA', 'miRNA.csv', 'MiDir_context.csv'), ('LncDrug', 'lncRNA', 'lncRNA.csv', 'LncDir_context.csv')]:
    rn = pd.read_csv(ROOT + '2.construct_network/' + rfile).iloc[:, 0].str.upper().tolist(); r2i = {n: i for i, n in enumerate(rn)}
    sub = raw[raw.ncRNA_Type == rtype].copy()
    sub['ri'] = sub.ncRNA_Name.str.upper().map(r2i); sub['di'] = sub.cid.map(lambda c: d2i.get(int(c)) if pd.notna(c) else None)
    sub = sub[sub.ri.notna() & sub.di.notna()].copy()
    sub['ri'] = sub.ri.astype(int); sub['di'] = sub.di.astype(int)
    c = pd.read_csv(ROOT + '13.context/' + ctx)
    # map (ncRNA upper, cid, effect) record -> disease list; context has one row per cell line
    c['key'] = list(zip(c.ncRNA.str.upper(), c.cid.astype(int), c.effect))
    dis = c.groupby('record').disease.apply(lambda s: sorted(set(x for x in s if isinstance(x, str)))).to_dict()
    ckey = c.groupby('record').key.first().to_dict()
    # align: the context file's record index enumerates the task's records in DR_Curated order
    sub = sub.reset_index(drop=True)
    sub['record'] = np.arange(len(sub))
    ok = sum(1 for i in range(len(sub)) if i in ckey and ckey[i] == (sub.ncRNA_Name.iloc[i].upper(), int(sub.cid.iloc[i]), sub.Effect.iloc[i]))
    print(task, 'records', len(sub), 'context rows', c.record.nunique(), 'aligned', ok)
    sub['diseases'] = sub.record.map(lambda i: '|'.join(dis.get(i, [])))
    sub['pheno'] = sub.Phenotype.fillna('').str.strip().str.lower()
    sub['cond'] = sub.Condition.fillna('').str.strip().str.lower()
    sub['ht'] = (sub.Throughput == 'High').astype(int)
    sub['year'] = pd.to_numeric(sub.Published_Year, errors='coerce')
    keep = sub[['ri', 'di', 'PMID', 'Effect', 'ht', 'Detection_Method', 'pheno', 'cond', 'diseases', 'year']].copy(); keep['task'] = task
    out.append(keep)
rec = pd.concat(out); rec.to_csv('research/exposure_next_20260917/records.csv', index=False)
print(rec.groupby('task').size(), rec.head())
