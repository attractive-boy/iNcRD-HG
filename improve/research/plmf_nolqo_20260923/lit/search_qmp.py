"""Positioning scan for query-masked message passing: exact edge deletion / unlearning / influence of
single edges on GNN states, leave-one-out GNN training, and target-link removal in pair-level predictors."""
import json, ssl, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from pathlib import Path
OUT = Path(__file__).resolve().parent
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
def get(url):
    for t in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 LitScan/1.2'})
            with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
                return r.read()
        except Exception:
            if t == 3: raise
            time.sleep(15 * (t + 1))
Q = {
 'edge_unlearning': '(all:unlearning) AND (all:edge OR all:edges) AND (all:"graph neural network" OR all:GNN)',
 'edge_influence': '(all:"influence function" OR all:"edge influence") AND (all:"graph neural network" OR all:GNN)',
 'loo_gnn': '(all:"leave-one-out") AND (all:"graph neural network" OR all:"message passing")',
 'target_removal_lp': '(all:"link prediction") AND (all:"remove" OR all:"removal" OR all:"exclude") AND (all:"target edge" OR all:"target link" OR all:"training edges")',
 'self_excluding': '(all:"self-excluding" OR all:"exclusion" OR all:"non-backtracking") AND (all:"message passing" OR all:"graph neural network")',
 'ease_loo': '(all:EASE OR all:"linear autoencoder") AND (all:"leave-one-out" OR all:"cross-validation" OR all:"diagonal")',
}
res = {}
for k, q in Q.items():
    url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': q, 'start': 0, 'max_results': 30, 'sortBy': 'relevance'})
    try:
        root = ET.fromstring(get(url)); ns = {'a': 'http://www.w3.org/2005/Atom'}
        res[k] = [dict(id=e.find('a:id', ns).text.split('/abs/')[-1], date=e.find('a:published', ns).text[:10], title=' '.join(e.find('a:title', ns).text.split()),
                       abstract=' '.join((e.find('a:summary', ns).text or '').split())[:1200]) for e in root.findall('a:entry', ns)]
        print(k, len(res[k]), flush=True)
    except Exception as ex:
        res[k] = dict(error=str(ex)); print(k, 'ERR', ex, flush=True)
    time.sleep(6)
json.dump(res, open(OUT / 'scan_qmp.json', 'w'), indent=1)
with open(OUT / 'scan_qmp_titles.txt', 'w') as f:
    for k, v in res.items():
        f.write(f'## {k}\n')
        if isinstance(v, dict): f.write(f'  ERROR {v}\n'); continue
        for it in v: f.write(f"  {it['date']} {it['id']} {it['title']}\n")
