"""Label-free literature-attention prior per miRNA: PubMed esearch counts (all / drug-response context)."""
import json, ssl, time, urllib.request, urllib.parse, csv, os, sys, re
import pandas as pd
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921')
from plmf_splits import load_dataset
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
names = set()
for ds in ('MDR', 'MDS'):
    mir, _, _, _ = load_dataset(ds); names |= set(mir['miRNA_Name'].astype(str))
def term_of(n):
    s = n.lower().replace('hsa-', '')
    s = re.sub(r'-(3p|5p)$', '', s)
    s = re.sub(r'-[12]$', '', s)          # miR-199a-1 -> miR-199a
    s = s.replace('mir-', 'miR-').replace('let-', 'let-')
    if s.startswith('let'):
        return f'"{s}"[tiab] OR "{s.replace("let-", "let")}"[tiab]'
    num = s.replace('miR-', '')
    return f'"miR-{num}"[tiab] OR "microRNA-{num}"[tiab] OR "miRNA-{num}"[tiab] OR "miR{num}"[tiab]'
def count(term, tries=4):
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&rettype=count&term=' + urllib.parse.quote(term)
    for t in range(tries):
        try:
            d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'incrd-hg/1.0'}), timeout=30, context=ctx).read())
            return int(d['esearchresult']['count'])
        except Exception as e:
            time.sleep(2 * (t + 1))
    return -1
out = 'data/pubmed_mirna_plmf.csv'; done = {}
if os.path.exists(out):
    done = {r['name']: r for r in csv.DictReader(open(out))}
fh = open(out, 'a', newline=''); w = csv.writer(fh)
if not done: w.writerow(['name', 'term', 'n_all', 'n_drugresp', 'n_cancer'])
import random; random.seed(4); _nm = sorted(names); random.shuffle(_nm)
for i, n in enumerate(_nm):
    if n in done: continue
    q = '(' + term_of(n) + ')'
    r = [count(q), count(q + ' AND (resistance[tiab] OR resistant[tiab] OR sensitivity[tiab] OR chemosensitivity[tiab] OR chemoresistance[tiab])'), count(q + ' AND (cancer[tiab] OR tumor[tiab] OR tumour[tiab] OR carcinoma[tiab])')]
    w.writerow([n, q, *r]); fh.flush()
    if i % 50 == 0: print(i, n, r, flush=True)
    time.sleep(0.35)
print('DONE')
