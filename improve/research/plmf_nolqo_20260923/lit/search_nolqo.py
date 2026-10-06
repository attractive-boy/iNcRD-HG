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
 'target_link': '(all:"link prediction") AND (all:"target link" OR all:"target edge" OR all:"dataset shift" OR all:FakeEdge OR all:"edge invariant")',
 'common_neighbor': '(all:"link prediction") AND (all:"common neighbor" OR all:"common neighbour" OR all:"neural common neighbor")',
 'path_counting': '(all:"simple path" OR all:"path counting" OR all:"cycle counting" OR all:"subgraph counting") AND (all:"graph neural network") AND (all:"link" OR all:"expressive")',
 'ease_linear_ae': '(all:EASE OR all:"linear autoencoder" OR all:"shallow autoencoder" OR all:"closed-form") AND (all:"collaborative filtering" OR all:recommendation) AND (all:item)',
 'identity_overfit': '(all:autoencoder OR all:"graph filter") AND (all:"identity" OR all:"diagonal") AND (all:"collaborative filtering" OR all:recommendation) AND (all:constraint OR all:dropout OR all:emphasis)',
 'masked_edge': '(all:"masked graph autoencoder" OR all:MaskGAE OR all:"edge masking" OR all:"masked edge") AND (all:"link prediction")',
 'bipartite_lp': '(all:bipartite) AND (all:"link prediction") AND (all:"graph neural network" OR all:GNN)',
 'loo_closed_form': '(all:"Sherman-Morrison" OR all:Woodbury OR all:"leave-one-out") AND (all:"link prediction" OR all:"collaborative filtering" OR all:"graph")',
 'heuristic_learning': '(all:"link prediction") AND (all:heuristic OR all:heuristics) AND (all:learnable OR all:"heuristic learning" OR all:"mixture")',
 'gnn_cf_training_free': '(all:"graph filter" OR all:"graph signal processing" OR all:"training-free") AND (all:"collaborative filtering")',
 'degree_bias_lp': '(all:"link prediction") AND (all:"degree bias" OR all:"popularity bias" OR all:"low-degree")',
 'pairwise_rep_2026': '(all:"link prediction") AND (all:"pairwise" OR all:"pair representation" OR all:"pair-wise") AND (all:"graph neural network")',
}
EPMC = {
 'mda_2026': '("miRNA-drug" OR "microRNA-drug" OR "miRNA drug" OR "microRNA drug") AND (association OR resistance OR sensitivity) AND prediction AND (PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'ncrna_drug_2026': '("ncRNA-drug" OR "lncRNA-drug" OR "circRNA-drug" OR "RNA-drug") AND (association OR resistance OR sensitivity) AND (prediction OR "graph neural") AND (PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'plmf_mda': '("PLMF-MDA" OR "language model" ) AND ("miRNA-drug" OR "microRNA-drug" OR "miRNA drug") AND (PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'mda_leakage': '(association prediction) AND ("data leakage" OR "information leakage" OR "target link") AND (miRNA OR lncRNA OR drug) AND ("graph neural" OR GNN) AND (PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'seed_family_assoc': '(miRNA OR microRNA) AND ("seed region" OR "seed sequence" OR "seed family" OR "miRNA family") AND ("association prediction" OR "link prediction") AND (PUB_YEAR:2023 OR PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'common_neighbor_bio': '("common neighbor" OR "common neighbour" OR "neural common neighbor") AND ("association prediction" OR "interaction prediction") AND (PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026)',
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
json.dump(results, open(OUT / 'scan_nolqo.json', 'w'), indent=1)
with open(OUT / 'scan_nolqo_titles.txt', 'w') as f:
    for k, v in results.items():
        f.write(f'## {k}\n')
        if isinstance(v, dict):
            f.write(f'  ERROR {v}\n'); continue
        for it in v:
            f.write(f"  {it['date']} {it['id']} {it['title']}\n")
print('done')
