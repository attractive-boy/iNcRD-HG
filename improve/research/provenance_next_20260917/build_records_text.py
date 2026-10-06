"""Same filtering/order as exposure_next_20260917/build_records.py, returning the raw frames with text columns."""
import numpy as np, pandas as pd
ROOT = '/data1/licheng/incrd-hg/incrd-run/'


def filtered_frames(raw):
    drugs = pd.read_csv(ROOT + '2.construct_network/Drug.csv'); d2i = {int(c): i for i, c in enumerate(drugs.CID)}
    out = []
    for task, rtype, rfile in [('MiDrug', 'miRNA', 'miRNA.csv'), ('LncDrug', 'lncRNA', 'lncRNA.csv')]:
        rn = pd.read_csv(ROOT + '2.construct_network/' + rfile).iloc[:, 0].str.upper().tolist(); r2i = {n: i for i, n in enumerate(rn)}
        sub = raw[raw.ncRNA_Type == rtype].copy()
        sub['ri'] = sub.ncRNA_Name.str.upper().map(r2i); sub['di'] = sub.cid.map(lambda c: d2i.get(int(c)) if pd.notna(c) else None)
        sub = sub[sub.ri.notna() & sub.di.notna()].copy()
        sub['ri'] = sub.ri.astype(int); sub['di'] = sub.di.astype(int)
        out.append((task, sub.reset_index(drop=True)))
    return out
