"""Frontier scan (round 7): architecture levers for a bipartite miRNA-drug GNN that need NO external
knowledge -- path/labeling-trick encoders, pair-adaptive attention, retrieval / in-context label
transfer, long-tail & cold nodes, train/test message-graph consistency, sequence (seed) encoders,
PU / unlabeled negatives.  arXiv export API (paced, retried) + Europe PMC core."""
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
        except Exception as ex:
            if t == tries - 1:
                raise
            time.sleep(20 * (t + 1))


ARXIV = {
 'nbfnet_path': '(all:"Bellman-Ford" OR all:"path-based" OR all:"conditional message passing" OR all:"labeling trick") AND all:"link prediction"',
 'pair_attention': '(all:"link prediction") AND (all:"pairwise encoding" OR all:"adaptive graph transformer" OR all:LPFormer OR all:"pair-wise attention")',
 'retrieval_lp': '(all:"retrieval" OR all:"in-context") AND (all:"link prediction" OR all:"graph neural network") AND (all:"label" OR all:"neighbors")',
 'longtail_lp': '(all:"link prediction" OR all:"recommendation") AND (all:"long-tail" OR all:"low-degree" OR all:"tail nodes" OR all:"cold-start nodes") AND all:graph',
 'train_test_msg': '(all:"link prediction") AND (all:"distribution shift" OR all:"train-test" OR all:"test-time" OR all:"message passing graph") AND all:GNN',
 'structural_encoding': '(all:"link prediction") AND (all:"structural encoding" OR all:"positional encoding" OR all:"random walk") AND all:2025',
 'moe_lp': '(all:"link prediction") AND (all:"mixture of experts" OR all:"gating") AND all:graph',
 'mirna_seq_encoder': '(all:miRNA OR all:microRNA) AND (all:"seed region" OR all:"sequence encoder" OR all:"language model") AND (all:association OR all:interaction)',
 'mda_2026': '(all:miRNA OR all:microRNA OR all:ncRNA) AND all:drug AND (all:resistance OR all:sensitivity OR all:association) AND all:prediction',
 'pu_lp': '(all:"positive-unlabeled" OR all:"PU learning" OR all:"hard negative") AND (all:"link prediction" OR all:"drug-target")',
 'bipartite_cf_gnn': '(all:"collaborative filtering" OR all:"bipartite graph") AND (all:"graph neural network") AND (all:"sparse" OR all:"cold") AND all:2026',
 'contextual_rank': '(all:"link prediction" OR all:"collaborative filtering") AND (all:"listwise" OR all:"set-aware" OR all:"contextual re-ranking" OR all:"calibration")',
}
EPMC = {
 'mda_models_2026': '("miRNA-drug" OR "microRNA-drug" OR "miRNA drug") AND (association OR resistance OR sensitivity) AND prediction AND (PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'ncrna_drug_2026': '("ncRNA-drug" OR "lncRNA-drug" OR "circRNA-drug") AND (association OR resistance OR sensitivity) AND prediction AND (PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'path_gnn_bio': '("NBFNet" OR "Bellman-Ford" OR "path-based" OR "labeling trick") AND ("link prediction" OR "association prediction") AND (PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'seed_family_mda': '(miRNA OR microRNA) AND ("seed region" OR "seed sequence" OR "miRNA family") AND ("graph neural" OR "deep learning") AND (association prediction) AND (PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026)',
}
results = {}
for k, q in ARXIV.items():
    try:
        url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': q, 'start': 0, 'max_results': 30, 'sortBy': 'submittedDate', 'sortOrder': 'descending'})
        root = ET.fromstring(get(url)); ns = {'a': 'http://www.w3.org/2005/Atom'}
        items = []
        for e in root.findall('a:entry', ns):
            items.append(dict(id=e.find('a:id', ns).text.split('/abs/')[-1], date=e.find('a:published', ns).text[:10],
                              title=' '.join(e.find('a:title', ns).text.split()), abstract=' '.join((e.find('a:summary', ns).text or '').split())[:1500]))
        results['arxiv:' + k] = items; print('arxiv', k, len(items), flush=True)
    except Exception as ex:
        results['arxiv:' + k] = dict(error=str(ex)); print('arxiv', k, 'ERR', ex, flush=True)
    time.sleep(15)
for k, q in EPMC.items():
    try:
        url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode({'query': q, 'format': 'json', 'pageSize': 40, 'resultType': 'core', 'sort': 'P_PDATE_D desc'})
        js = json.loads(get(url)); items = []
        for r in js.get('resultList', {}).get('result', []):
            items.append(dict(pmid=r.get('pmid'), date=r.get('firstPublicationDate'), journal=r.get('journalTitle'), title=r.get('title'),
                              abstract=(r.get('abstractText') or '')[:1500]))
        results['epmc:' + k] = items; print('epmc', k, len(items), flush=True)
    except Exception as ex:
        results['epmc:' + k] = dict(error=str(ex)); print('epmc', k, 'ERR', ex, flush=True)
    time.sleep(1)
json.dump(results, open(OUT / 'scan_internal.json', 'w'), indent=1, ensure_ascii=False)
with open(OUT / 'scan_internal_titles.txt', 'w') as fh:
    for k, v in results.items():
        fh.write(f'## {k}\n')
        if isinstance(v, dict):
            fh.write(f'  ERROR {v}\n'); continue
        for it in v:
            fh.write(f"  {it.get('date')} {it.get('id') or it.get('pmid')} {it.get('title')}\n")
print('done')
