"""Novelty / prior-art scan for external-relation channels on miRNA-drug association prediction (arXiv + Europe PMC)."""
import json, ssl, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from pathlib import Path
OUT = Path(__file__).resolve().parent
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def get(url, timeout=40):
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0 LitScan/1.0 (mailto:lit@example.org)'})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r: return r.read()
ARXIV = {
 'mirna_drug_disease': 'all:miRNA AND all:drug AND all:disease AND (all:association OR all:resistance) AND all:prediction',
 'pgx_mirna': '(all:pharmacogenomic OR all:"drug response" OR all:GDSC) AND all:miRNA AND all:prediction',
 'cerna_drug': '(all:ceRNA OR all:"competing endogenous") AND all:drug AND all:prediction',
 'lincs_assoc': '(all:LINCS OR all:"connectivity map" OR all:"perturbation signature") AND (all:miRNA OR all:ncRNA) AND all:association',
 'sparse_lp_side': 'all:"link prediction" AND (all:"side information" OR all:"external knowledge" OR all:"knowledge-enhanced") AND (all:"low-degree" OR all:"cold" OR all:"sparse")',
 'multirel_heuristic': '(all:"multi-relational" OR all:"heterogeneous") AND all:"link prediction" AND (all:heuristic OR all:"meta-path") AND all:2026',
 'genomic_cluster_mirna': 'all:miRNA AND all:cluster AND (all:"genomic" OR all:"polycistronic") AND all:prediction',
 'kg_mda': '(all:"knowledge graph") AND all:miRNA AND all:drug',
}
EPMC = {
 'mda_disease_tripartite': '(miRNA OR microRNA) AND drug AND disease AND (tripartite OR "heterogeneous network" OR "knowledge graph") AND (association prediction OR "resistance prediction") AND (PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'mda_gene_target': '(miRNA OR microRNA) AND drug AND ("target gene" OR miRTarBase OR "gene set") AND (resistance OR sensitivity) AND (association prediction) AND (PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'mda_gdsc': '(miRNA OR microRNA) AND (GDSC OR CCLE OR pharmacogenomic) AND (drug resistance OR drug sensitivity) AND prediction AND (PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'mda_lincs': '(miRNA OR microRNA) AND (LINCS OR "connectivity map" OR CMap) AND drug AND (association OR resistance) AND prediction AND (PUB_YEAR:2023 OR PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'cerna_drug_pred': '(ceRNA OR "competing endogenous RNA" OR "lncRNA-miRNA") AND drug AND (resistance OR sensitivity) AND (prediction OR predicting) AND (graph OR network) AND (PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'mirna_cluster_pred': '(miRNA OR microRNA) AND ("genomic cluster" OR "miRNA cluster" OR polycistronic) AND association AND prediction AND (PUB_YEAR:2023 OR PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'ncrna_drug_2026_latest': '(miRNA OR ncRNA OR lncRNA OR circRNA) AND drug AND (resistance OR sensitivity) AND association AND prediction AND (PUB_YEAR:2026)',
 'plmf_cites': '"PLMF-MDA" OR (CITES:(41045000_MED))',
}
results = {}
for k,q in ARXIV.items():
    url = 'http://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': q, 'start':0, 'max_results': 40, 'sortBy':'submittedDate', 'sortOrder':'descending'})
    try:
        raw = get(url); root = ET.fromstring(raw); ns={'a':'http://www.w3.org/2005/Atom'}; rows=[]
        for e in root.findall('a:entry', ns):
            rows.append({'id': e.find('a:id',ns).text.strip(), 'title': ' '.join(e.find('a:title',ns).text.split()), 'published': e.find('a:published',ns).text[:10], 'summary': ' '.join(e.find('a:summary',ns).text.split())[:900]})
        results['arxiv:'+k]=rows; print('arxiv',k,len(rows),flush=True)
    except Exception as ex: results['arxiv:'+k]={'error':str(ex)}; print('arxiv',k,'ERR',ex,flush=True)
    time.sleep(3.2)
for k,q in EPMC.items():
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode({'query': q, 'format':'json','pageSize':60,'sort':'P_PDATE_D desc','resultType':'core'})
    try:
        d=json.loads(get(url)); rows=[]
        for r in d.get('resultList',{}).get('result',[]):
            rows.append({'pmid': r.get('pmid'), 'title': r.get('title'), 'journal': r.get('journalTitle') or (r.get('journalInfo',{}) or {}).get('journal',{}).get('title'), 'year': r.get('pubYear'), 'abstract': (r.get('abstractText') or '')[:1500]})
        results['epmc:'+k]=rows; print('epmc',k,len(rows),d.get('hitCount'),flush=True)
    except Exception as ex: results['epmc:'+k]={'error':str(ex)}; print('epmc',k,'ERR',ex,flush=True)
    time.sleep(1.0)
json.dump(results, open(OUT/'scan_xrel.json','w'), indent=1, ensure_ascii=False)
with open(OUT/'scan_xrel_titles.txt','w') as f:
    for k,v in results.items():
        f.write(f'### {k}\n')
        if isinstance(v, list):
            for r in v: f.write(f"- [{r.get('published') or r.get('year')}] {r.get('title')} ({r.get('id') or r.get('pmid')}; {r.get('journal','')})\n")
print('DONE')
