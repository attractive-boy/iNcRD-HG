"""Round-10 frontier scan: PU / all-unobserved supervision for link prediction, negative sampling for
bipartite GNNs, within-node ranking / popularity, conditional (hyper-network) decoders, neighbour-set
transformers, and 2025-26 miRNA/ncRNA-drug models.  arXiv export API + Europe PMC."""
import json, ssl, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from pathlib import Path
OUT = Path(__file__).resolve().parent
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
def get(url, timeout=60, tries=4):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 LitScan/1.3 (mailto:lit@example.org)'})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r: return r.read()
        except Exception:
            if t == tries - 1: raise
            time.sleep(15 * (t + 1))
ARXIV = {
 'pu_linkpred': '(all:"positive-unlabeled" OR all:"PU learning") AND (all:"link prediction" OR all:"graph")',
 'neg_sampling_gnn': '(all:"negative sampling") AND (all:"link prediction" OR all:"graph neural network") AND (all:"hard negative" OR all:"distribution" OR all:"bias")',
 'whole_matrix': '(all:"non-sampling" OR all:"whole-data" OR all:"all unobserved" OR all:"full softmax" OR all:"efficient non-sampling") AND (all:recommendation OR all:"collaborative filtering" OR all:"graph")',
 'hypernet_decoder': '(all:hypernetwork OR all:"conditional decoder" OR all:"FiLM") AND (all:"link prediction" OR all:"collaborative filtering" OR all:"drug")',
 'set_transformer_lp': '(all:"neighborhood" OR all:"neighbor set") AND (all:transformer OR all:"cross-attention") AND (all:"link prediction")',
 'popularity_within': '(all:"popularity bias" OR all:"degree bias") AND (all:"link prediction" OR all:"collaborative filtering") AND (all:2025 OR all:2026 OR all:debias)',
 'lp_2026': '(all:"link prediction") AND (all:"graph neural network") AND (all:"common neighbor" OR all:"heuristic" OR all:"pairwise")',
 'label_trick_residual': '(all:"correct and smooth" OR all:"residual propagation" OR all:"label propagation") AND (all:"link prediction" OR all:"collaborative filtering")',
 'ssl_lp': '(all:"masked autoencoder" OR all:"self-supervised") AND (all:"link prediction") AND (all:bipartite OR all:"recommendation")',
}
EPMC = {
 'mda_2026': '("miRNA-drug" OR "microRNA-drug" OR "miRNA drug") AND (association OR resistance OR sensitivity) AND prediction AND (PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'ncrna_drug_pu': '("ncRNA" OR "miRNA" OR "lncRNA" OR "circRNA") AND ("drug" OR "disease") AND ("positive-unlabeled" OR "negative sampling" OR "unlabeled") AND (PUB_YEAR:2025 OR PUB_YEAR:2026)',
}
results = {}
for k, q in ARXIV.items():
    try:
        url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': q, 'start': 0, 'max_results': 40, 'sortBy': 'submittedDate', 'sortOrder': 'descending'})
        root = ET.fromstring(get(url)); ns = {'a': 'http://www.w3.org/2005/Atom'}
        items = [dict(id=e.find('a:id', ns).text.split('/abs/')[-1], date=e.find('a:published', ns).text[:10],
                      title=' '.join(e.find('a:title', ns).text.split()), abstract=' '.join((e.find('a:summary', ns).text or '').split())[:1500]) for e in root.findall('a:entry', ns)]
        results['arxiv:' + k] = items; print('arxiv', k, len(items), flush=True)
    except Exception as ex:
        results['arxiv:' + k] = dict(error=str(ex)); print('arxiv', k, 'ERR', ex, flush=True)
    time.sleep(6)
for k, q in EPMC.items():
    try:
        url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode({'query': q, 'format': 'json', 'pageSize': 50, 'resultType': 'core', 'sort': 'P_PDATE_D desc'})
        js = json.loads(get(url))
        items = [dict(id=r.get('pmid') or r.get('doi') or r.get('id'), date=r.get('firstPublicationDate', ''), title=r.get('title', ''), abstract=(r.get('abstractText') or '')[:1500]) for r in js.get('resultList', {}).get('result', [])]
        results['epmc:' + k] = items; print('epmc', k, len(items), flush=True)
    except Exception as ex:
        results['epmc:' + k] = dict(error=str(ex)); print('epmc', k, 'ERR', ex, flush=True)
    time.sleep(2)
json.dump(results, open(OUT / 'scan_r10.json', 'w'), indent=1)
with open(OUT / 'scan_r10_titles.txt', 'w') as f:
    for k, v in results.items():
        f.write(f'### {k}\n')
        for it in (v if isinstance(v, list) else []):
            f.write(f"{it['date']} {it['id']} {it['title']}\n")
