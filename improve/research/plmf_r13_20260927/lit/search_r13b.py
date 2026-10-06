"""Round-13 positioning scan for the entity random-effects head and the negative-pool audit."""
import json, ssl, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from pathlib import Path
OUT = Path(__file__).resolve().parent
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
def get(url, timeout=60, tries=4):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 LitScan/1.5 (mailto:lit@example.org)'})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r: return r.read()
        except Exception:
            if t == tries - 1: raise
            time.sleep(15 * (t + 1))
ARXIV = {
 'random_effects_nn': '(all:"random effects" OR all:"mixed effects" OR all:"mixed-effects") AND (all:"neural network" OR all:"deep learning") AND (all:"high-cardinality" OR all:"categorical" OR all:"recommendation" OR all:"link")',
 'glmm_graph': '(all:"random effects" OR all:"mixed model" OR all:"empirical Bayes") AND (all:"graph neural network" OR all:"link prediction" OR all:"collaborative filtering")',
 'bias_terms_cf': '(all:"user bias" OR all:"item bias" OR all:"bias term") AND (all:"collaborative filtering" OR all:"matrix factorization" OR all:"graph neural network")',
 'post_hoc_calib_lp': '(all:"post-hoc calibration" OR all:"recalibration" OR all:"calibration") AND (all:"link prediction" OR all:"recommendation" OR all:"graph neural network")',
 'negative_sampling_leak': '(all:"negative sampling" OR all:"negative edges") AND (all:"leakage" OR all:"test edges" OR all:"evaluation") AND (all:"link prediction")',
}
EPMC = {
 'mda_eval_leak': '("negative sampling" OR "negative samples") AND ("association prediction" OR "link prediction") AND (leakage OR bias OR "evaluation") AND (PUB_YEAR:2025 OR PUB_YEAR:2026)',
}
results = {}
for k, q in ARXIV.items():
    try:
        url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': q, 'start': 0, 'max_results': 25, 'sortBy': 'relevance', 'sortOrder': 'descending'})
        root = ET.fromstring(get(url)); ns = {'a': 'http://www.w3.org/2005/Atom'}
        items = [dict(id=e.find('a:id', ns).text.split('/abs/')[-1], date=e.find('a:published', ns).text[:10],
                      title=' '.join(e.find('a:title', ns).text.split()), abstract=' '.join((e.find('a:summary', ns).text or '').split())[:1500]) for e in root.findall('a:entry', ns)]
        results['arxiv:' + k] = items; print('arxiv', k, len(items), flush=True)
    except Exception as ex:
        results['arxiv:' + k] = dict(error=str(ex)); print('arxiv', k, 'ERR', ex, flush=True)
    time.sleep(6)
for k, q in EPMC.items():
    try:
        url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode({'query': q, 'format': 'json', 'pageSize': 40, 'resultType': 'core', 'sort': 'P_PDATE_D desc'})
        js = json.loads(get(url))
        items = [dict(id=r.get('pmid') or r.get('doi') or r.get('id'), date=r.get('firstPublicationDate', ''), title=r.get('title', ''), abstract=(r.get('abstractText') or '')[:1500]) for r in js.get('resultList', {}).get('result', [])]
        results['epmc:' + k] = items; print('epmc', k, len(items), flush=True)
    except Exception as ex:
        results['epmc:' + k] = dict(error=str(ex)); print('epmc', k, 'ERR', ex, flush=True)
json.dump(results, open(OUT / 'scan_r13b.json', 'w'), indent=1)
with open(OUT / 'scan_r13b_titles.txt', 'w') as f:
    for k, v in results.items():
        f.write(f'### {k}\n')
        for it in (v if isinstance(v, list) else []):
            f.write(f"{it['date']} {it['id']} {it['title']}\n")
print('done')
