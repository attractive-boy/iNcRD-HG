"""Frontier scan (round 8, no-LQO): target-link leakage remedies without leave-out training,
common-neighbour / path-counting link predictors, closed-form linear autoencoders (EASE family),
masked-edge objectives, bipartite link prediction, Woodbury/leave-one-out closed forms, and 2025-26
miRNA/ncRNA-drug association models.  arXiv export API (paced) + Europe PMC core."""
import json, ssl, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from pathlib import Path
OUT = Path(__file__).resolve().parent
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE


def get(url, timeout=60, tries=4):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 LitScan/1.2 (mailto:lit@example.org)'})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                return r.read()
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(15 * (t + 1))


ARXIV = {
 'degree_tail': '(all:"tail node" OR all:"low-degree" OR all:"degree bias" OR all:"degree-specific") AND (all:"graph neural network")',
 'jk_residual': '(all:"jumping knowledge" OR all:"initial residual" OR all:"layer aggregation" OR all:"multi-scale") AND (all:"graph neural network")',
 'semantic_attention': '(all:heterogeneous) AND (all:"relation attention" OR all:"semantic attention" OR all:"relation-aware" OR all:"type-aware") AND (all:"graph neural network")',
 'feature_structure_mix': '(all:"feature propagation" OR all:"content" OR all:"attribute") AND (all:"structure") AND (all:"adaptive" OR all:"gate" OR all:"gating") AND (all:"link prediction" OR all:"cold start")',
 'hollow_encoder': '(all:"edge dropping" OR all:"DropEdge" OR all:"self-loop" OR all:"leakage") AND (all:"link prediction") AND (all:"encoder")',
 'normalization_gnn': '(all:"degree normalization" OR all:"aggregation normalization" OR all:"mean aggregation" OR all:"sum aggregation") AND (all:"graph neural network") AND (all:"link" OR all:"expressive")',
 'bilinear_decoder': '(all:"bilinear" OR all:"DistMult" OR all:"Hadamard") AND (all:"link prediction") AND (all:"decoder") AND (all:"graph")',
 'heterophily_bip': '(all:bipartite) AND (all:"graph convolution" OR all:"message passing") AND (all:"normalization" OR all:"degree" OR all:"popularity")',
}
EPMC = {
 'mda_2026': '("miRNA-drug" OR "microRNA-drug" OR "miRNA drug") AND (association OR resistance OR sensitivity) AND prediction AND (PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'ncrna_encoder': '("ncRNA-drug" OR "miRNA-disease" OR "lncRNA-disease" OR "circRNA-drug") AND ("encoder" OR "graph neural") AND ("attention" OR "gating" OR "jumping" OR "residual") AND (PUB_YEAR:2025 OR PUB_YEAR:2026)',
}
results = {}
for k, q in ARXIV.items():
    try:
        url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': q, 'start': 0, 'max_results': 40, 'sortBy': 'relevance', 'sortOrder': 'descending'})
        root = ET.fromstring(get(url)); ns = {'a': 'http://www.w3.org/2005/Atom'}
        items = []
        for e in root.findall('a:entry', ns):
            items.append(dict(id=e.find('a:id', ns).text.split('/abs/')[-1], date=e.find('a:published', ns).text[:10],
                              title=' '.join(e.find('a:title', ns).text.split()), abstract=' '.join((e.find('a:summary', ns).text or '').split())[:1800]))
        results['arxiv:' + k] = items; print('arxiv', k, len(items), flush=True)
    except Exception as ex:
        results['arxiv:' + k] = dict(error=str(ex)); print('arxiv', k, 'ERR', ex, flush=True)
    time.sleep(6)
for k, q in EPMC.items():
    try:
        url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode({'query': q, 'format': 'json', 'pageSize': 50, 'resultType': 'core', 'sort': 'P_PDATE_D desc'})
        js = json.loads(get(url)); items = []
        for r in js.get('resultList', {}).get('result', []):
            items.append(dict(id=r.get('pmid') or r.get('doi') or r.get('id'), date=r.get('firstPublicationDate', ''), journal=r.get('journalTitle') or (r.get('journalInfo', {}) or {}).get('journal', {}).get('title', ''),
                              title=r.get('title', ''), abstract=(r.get('abstractText') or '')[:1800]))
        results['epmc:' + k] = items; print('epmc', k, len(items), flush=True)
    except Exception as ex:
        results['epmc:' + k] = dict(error=str(ex)); print('epmc', k, 'ERR', ex, flush=True)
    time.sleep(2)
json.dump(results, open(OUT / 'scan_enc.json', 'w'), indent=1)
with open(OUT / 'scan_enc_titles.txt', 'w') as f:
    for k, v in results.items():
        f.write(f'## {k}\n')
        if isinstance(v, dict):
            f.write(f'  ERROR {v}\n'); continue
        for it in v:
            f.write(f"  {it['date']} {it['id']} {it['title']}\n")
print('done')
