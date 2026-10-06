# -*- coding: utf-8 -*-
"""Fetch CLIP-supported miRNA-target edges from ENCORI (starBase v3 API) for the
union of PLMF-MDA MDR/MDS miRNAs.  MPHGNN's miRNA-gene relation came from ENCORI,
so this is the same source.  One request per miRNA (miRNA=all is refused by the
API); the raw rows (geneID, geneName, clipExpNum, degraExpNum, program votes)
are stored so the edge filter can be chosen offline."""
import json, os, ssl, sys, time, urllib.request, urllib.parse
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PLMF = '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/new5_repro_20260919/repos/PLMF-MDA-main/data/unpacked2/data'
OUT = os.path.join(HERE, 'cache', 'encori_targets.tsv.gz')
STATE = os.path.join(HERE, 'cache', 'encori_state.json')
BASE = ('https://rnasysu.com/encori/api/miRNATarget/?assembly=hg38&geneType=mRNA&clipExpNum=1'
        '&degraExpNum=0&pancancerNum=0&programNum=1&program=None&target=all&cellType=all&miRNA=')
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
KEEP = ['miRNAid', 'miRNAname', 'geneID', 'geneName', 'geneType', 'clipExpNum', 'degraExpNum',
        'PITA', 'RNA22', 'miRmap', 'microT', 'miRanda', 'PicTar', 'TargetScan']


def fetch(mi, tries=3):
    for i in range(tries):
        try:
            r = urllib.request.urlopen(urllib.request.Request(BASE + urllib.parse.quote(mi), headers={'User-Agent': 'Mozilla/5.0'}),
                                       timeout=120, context=CTX)
            txt = r.read().decode('utf-8', 'ignore')
            lines = [l for l in txt.splitlines() if l and not l.startswith('#')]
            if not lines:
                return 'empty', []
            if 'parameter' in lines[0] and 'miRNA' in lines[0]:
                return 'not_available', []
            hdr = lines[0].split('\t')
            idx = [hdr.index(c) for c in KEEP if c in hdr]
            rows = []
            for l in lines[1:]:
                f = l.split('\t')
                if len(f) >= len(hdr):
                    rows.append([f[i] for i in idx])
            return 'ok', rows
        except Exception as e:
            err = repr(e)[:120]
            time.sleep(2 * (i + 1))
    return 'error:' + err, []


def main():
    names = set()
    for ds, suf in (('MDR', 'R'), ('MDS', 'S')):
        names |= set(pd.read_excel(f'{PLMF}/{ds}/miRNA_ID_{suf}.xlsx')['miRNA_Name'].astype(str))
    names = sorted(names)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    rows_all = []
    if os.path.exists(OUT):
        rows_all = pd.read_csv(OUT, sep='\t', dtype=str).values.tolist()
    todo = [n for n in names if n not in state]
    print('miRNAs', len(names), 'todo', len(todo), flush=True)
    t0 = time.time()
    for i, mi in enumerate(todo):
        status, rows = fetch(mi)
        state[mi] = dict(status=status, n=len(rows))
        rows_all.extend(rows)
        if (i + 1) % 50 == 0 or i == len(todo) - 1:
            pd.DataFrame(rows_all, columns=KEEP).to_csv(OUT, sep='\t', index=False, compression='gzip')
            json.dump(state, open(STATE, 'w'))
            n_ok = sum(v['status'] == 'ok' for v in state.values())
            print(f'{i + 1}/{len(todo)} done, ok {n_ok}, rows {len(rows_all)}, {time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(rows_all, columns=KEEP)
    df.to_csv(OUT, sep='\t', index=False, compression='gzip')
    json.dump(state, open(STATE, 'w'))
    st = pd.Series([v['status'].split(':')[0] for v in state.values()]).value_counts().to_dict()
    print('status counts', st, 'total rows', len(df), 'miRNAs with rows', df.miRNAname.nunique(), 'genes', df.geneID.nunique())


if __name__ == '__main__':
    main()
