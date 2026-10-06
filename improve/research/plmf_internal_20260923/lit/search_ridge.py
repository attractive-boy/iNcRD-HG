"""Novelty check for the closed-form collaborative ridge layer (EASE / side-information EASE / differentiable
closed-form solvers) inside GNN link predictors, and for MDA/ncRNA uses."""
import json, ssl, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from pathlib import Path
OUT = Path(__file__).resolve().parent
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
def get(url, timeout=60, tries=4):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 LitScan/1.1 (mailto:lit@example.org)'})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                return r.read()
        except Exception:
            if t == tries - 1: raise
            time.sleep(20 * (t + 1))
ARXIV = {
 'ease_gnn': '(all:EASE OR all:"shallow autoencoder" OR all:"closed-form") AND (all:"graph neural network" OR all:"link prediction")',
 'ease_side': '(all:EASE OR all:"linear autoencoder") AND (all:"side information" OR all:"side-information" OR all:"cold-start")',
 'diff_ridge': '(all:"differentiable" AND (all:"ridge regression" OR all:"closed-form solver")) AND (all:graph OR all:recommendation OR all:"few-shot")',
 'closedform_cf_2025': '(all:"closed-form" OR all:"training-free") AND all:"collaborative filtering" AND (all:2025 OR all:2026)',
}
EPMC = {
 'ease_bio': '("EASE" OR "shallow autoencoder" OR "closed-form" OR "ridge regression") AND ("miRNA" OR "ncRNA" OR "lncRNA" OR "circRNA" OR "drug") AND ("association prediction" OR "link prediction")',
 'linear_ae_mda': '("linear autoencoder" OR "item-item" OR "SLIM" OR "EASE") AND (miRNA OR lncRNA OR circRNA OR drug) AND (association OR interaction) AND prediction',
}
res = {}
for k, q in ARXIV.items():
    try:
        url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': q, 'start': 0, 'max_results': 30, 'sortBy': 'relevance'})
        root = ET.fromstring(get(url)); ns = {'a': 'http://www.w3.org/2005/Atom'}
        res['arxiv:' + k] = [dict(id=e.find('a:id', ns).text.split('/abs/')[-1], date=e.find('a:published', ns).text[:10], title=' '.join(e.find('a:title', ns).text.split()), abstract=' '.join((e.find('a:summary', ns).text or '').split())[:1000]) for e in root.findall('a:entry', ns)]
        print('arxiv', k, len(res['arxiv:' + k]), flush=True)
    except Exception as ex:
        res['arxiv:' + k] = dict(error=str(ex)); print('arxiv', k, 'ERR', ex, flush=True)
    time.sleep(15)
for k, q in EPMC.items():
    try:
        url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode({'query': q, 'format': 'json', 'pageSize': 30, 'resultType': 'core'})
        js = json.loads(get(url))
        res['epmc:' + k] = [dict(pmid=r.get('pmid'), date=r.get('firstPublicationDate'), journal=r.get('journalTitle'), title=r.get('title'), abstract=(r.get('abstractText') or '')[:1000]) for r in js.get('resultList', {}).get('result', [])]
        print('epmc', k, len(res['epmc:' + k]), flush=True)
    except Exception as ex:
        res['epmc:' + k] = dict(error=str(ex)); print('epmc', k, 'ERR', ex, flush=True)
json.dump(res, open(OUT / 'scan_ridge.json', 'w'), indent=1, ensure_ascii=False)
with open(OUT / 'scan_ridge_titles.txt', 'w') as fh:
    for k, v in res.items():
        fh.write(f'## {k}\n')
        if isinstance(v, dict): fh.write(f'  ERROR {v}\n'); continue
        for it in v: fh.write(f"  {it.get('date')} {it.get('id') or it.get('pmid')} {it.get('title')}\n")
print('done')
