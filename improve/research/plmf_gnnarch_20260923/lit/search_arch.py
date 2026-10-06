"""Frontier scan: link-prediction architecture ideas relevant to a hetero GNN on PLMF-MDA (arXiv export + Europe PMC)."""
import json, ssl, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from pathlib import Path
OUT = Path(__file__).resolve().parent
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def get(url, timeout=40):
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0 LitScan/1.0 (mailto:lit@example.org)'})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r: return r.read()
ARXIV = {
 'target_edge_removal': '(all:"link prediction") AND (all:"target edge" OR all:"target link" OR all:"edge leakage" OR all:"message passing edges") AND all:GNN',
 'ncn_common_neighbor': '(all:"link prediction") AND (all:"common neighbor" OR all:"common neighbour") AND all:neural',
 'pairwise_structural_lp': '(all:"link prediction") AND (all:"pairwise" OR all:"structural features") AND (all:"heuristic" OR all:"subgraph") AND all:2025',
 'hypergraph_star_expansion': '(all:hypergraph) AND (all:"star expansion" OR all:"bipartite expansion") AND (all:"neural network" OR all:GNN) AND all:link',
 'side_info_hetero_lp': '(all:"heterogeneous graph") AND (all:"link prediction") AND (all:"side information" OR all:"auxiliary node" OR all:"knowledge") AND (all:"low-degree" OR all:sparse OR all:"cold")',
 'bipartite_lp_gnn': '(all:bipartite) AND (all:"link prediction") AND all:GNN AND (all:2025 OR all:2026)',
 'lp_foundation_2026': '(all:"link prediction") AND (all:"foundation model" OR all:"graph transformer") AND all:2026',
 'mda_gnn_2026': '(all:miRNA OR all:microRNA) AND all:drug AND all:association AND (all:"graph neural" OR all:hypergraph)',
}
EPMC = {
 'ncrna_drug_gnn_2026': '(miRNA OR microRNA OR ncRNA) AND drug AND (resistance OR sensitivity OR association) AND prediction AND ("graph neural" OR hypergraph OR "heterogeneous graph") AND (PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'provenance_node_gnn': '(publication OR PMID OR literature OR provenance) AND ("graph neural network" OR "heterogeneous graph") AND ("association prediction" OR "link prediction") AND (PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'hmdd_disease_bridge_mda': '(miRNA OR microRNA) AND drug AND disease AND ("heterogeneous graph" OR "knowledge graph" OR tripartite) AND association AND prediction AND (PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'common_neighbor_bio_lp': '("common neighbor" OR "common neighbour" OR "neural common") AND ("link prediction" OR "association prediction") AND (PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026)',
}
results = {}
for k,q in ARXIV.items():
    try:
        url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': q, 'start': 0, 'max_results': 25, 'sortBy': 'submittedDate', 'sortOrder': 'descending'})
        root = ET.fromstring(get(url)); ns = {'a':'http://www.w3.org/2005/Atom'}
        items = []
        for e in root.findall('a:entry', ns):
            items.append(dict(id=e.find('a:id', ns).text.split('/abs/')[-1], date=e.find('a:published', ns).text[:10], title=' '.join(e.find('a:title', ns).text.split()), abstract=' '.join((e.find('a:summary', ns).text or '').split())[:1200]))
        results['arxiv:'+k] = items; print('arxiv', k, len(items), flush=True)
    except Exception as ex:
        results['arxiv:'+k] = dict(error=str(ex)); print('arxiv', k, 'ERR', ex, flush=True)
    time.sleep(12)
for k,q in {}.items():
    try:
        url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode({'query': q, 'format': 'json', 'pageSize': 25, 'resultType': 'core', 'sort': 'P_PDATE_D desc'})
        js = json.loads(get(url)); items = []
        for r in js.get('resultList', {}).get('result', []):
            items.append(dict(pmid=r.get('pmid'), date=r.get('firstPublicationDate'), journal=r.get('journalTitle'), title=r.get('title'), abstract=(r.get('abstractText') or '')[:1200]))
        results['epmc:'+k] = items; print('epmc', k, len(items), flush=True)
    except Exception as ex:
        results['epmc:'+k] = dict(error=str(ex)); print('epmc', k, 'ERR', ex, flush=True)
    time.sleep(1)
json.dump(results, open(OUT/'scan_arch_arxiv.json','w'), indent=1, ensure_ascii=False)
with open(OUT/'scan_arch_arxiv_titles.txt','w') as fh:
    for k,v in results.items():
        fh.write(f'## {k}\n')
        if isinstance(v, dict): fh.write(f'  ERROR {v}\n'); continue
        for it in v: fh.write(f"  {it.get('date')} {it.get('id') or it.get('pmid')} {it.get('title')}\n")
print('done')
