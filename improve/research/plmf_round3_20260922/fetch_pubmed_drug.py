"""Label-free literature-attention prior per drug: PubMed esearch counts for the drug's preferred name."""
import json, ssl, time, urllib.request, urllib.parse, csv, os, sys
import numpy as np
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
dt = json.load(open('/data1/licheng/incrd-hg/baselines/MPHGNN_on_PLMF-MDA/cache/drug_targets.json'))['drugs']
ids = set()
for ds in ('MDR', 'MDS'):
    g = np.load(f'/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921/cache/graph_{ds}.npz', allow_pickle=True)
    ids |= set(g['drug_ids'].tolist())
def count(term, tries=4):
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&rettype=count&term=' + urllib.parse.quote(term)
    for t in range(tries):
        try:
            d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'incrd-hg/1.0'}), timeout=30, context=ctx).read())
            return int(d['esearchresult']['count'])
        except Exception as e:
            time.sleep(2 * (t + 1))
    return -1
out = 'data/pubmed_drug_plmf.csv'; done = {}
if os.path.exists(out):
    done = {r['drugbank_id']: r for r in csv.DictReader(open(out))}
fh = open(out, 'a', newline=''); w = csv.writer(fh)
if not done: w.writerow(['drugbank_id', 'name', 'n_all', 'n_mirna', 'n_resist', 'n_cancer'])
for i, db in enumerate(sorted(ids)):
    if db in done: continue
    name = (dt.get(db) or {}).get('pref_name') or ''
    if not name: w.writerow([db, '', -1, -1, -1, -1]); fh.flush(); continue
    q = f'"{name}"[tiab]'
    r = [count(q), count(q + ' AND (miRNA[tiab] OR microRNA[tiab] OR miR[tiab])'), count(q + ' AND (resistance[tiab] OR resistant[tiab] OR sensitivity[tiab])'), count(q + ' AND (cancer[tiab] OR tumor[tiab] OR tumour[tiab] OR carcinoma[tiab])')]
    w.writerow([db, name] + r); fh.flush(); print(i, db, name, r, flush=True); time.sleep(0.4)
