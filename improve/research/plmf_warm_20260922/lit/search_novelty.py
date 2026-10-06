"""Prior-art check for (a) curation-provenance / publication hyperedges in association prediction,
(b) sparse / low-degree entities inside warm link-prediction benchmarks."""
import json, ssl, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from pathlib import Path
OUT = Path(__file__).resolve().parent
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def get(url, timeout=40):
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0 LitScan/1.0 (mailto:lit@example.org)'})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r: return r.read()
ARXIV = {
 'prov_hyper': '(all:provenance OR all:"publication" OR all:"co-citation" OR all:"study-level") AND all:hyperedge AND (all:"link prediction" OR all:biomedical OR all:"knowledge graph")',
 'pmid_feature': '(all:PMID OR all:"literature evidence" OR all:"evidence source") AND (all:"association prediction" OR all:"link prediction") AND all:biomedical',
 'lowdeg_lp': '(all:"low-degree" OR all:"long-tail" OR all:"few-shot entities" OR all:"sparse nodes") AND all:"link prediction"',
 'longtail_kg': '(all:"long-tail" OR all:"few-shot") AND (all:"knowledge graph completion" OR all:"entity") AND all:2025',
 'degree_strat_eval': '(all:"degree-stratified" OR all:"per-degree" OR all:"stratified evaluation") AND (all:"link prediction" OR all:recommendation)',
 'mirna_family_prior': '(all:"miRNA family" OR all:"seed sequence") AND all:prediction AND (all:drug OR all:disease)',
}
EPMC = {
 'prov_hyper_bio': '(hypergraph OR hyperedge) AND (PMID OR publication OR "literature-derived" OR provenance) AND (association prediction OR "link prediction") AND (PUB_YEAR:2022 OR PUB_YEAR:2023 OR PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'evidence_weight_assoc': '("evidence" AND ("number of publications" OR "PubMed counts" OR "literature count")) AND (miRNA OR lncRNA OR drug) AND (association prediction)',
 'sparse_mirna_warm': '(miRNA OR lncRNA) AND (association prediction) AND ("low-degree" OR "sparse" OR "long-tail" OR "few known associations" OR "isolated nodes")',
 'ncrnadrug_db': '"ncRNADrug" OR "NoncoRNA" OR "ncDR" AND (database) AND (drug resistance)',
 'family_prior_mda': '(miRNA) AND ("family" OR "seed") AND (drug OR disease) AND (association prediction) AND (PUB_YEAR:2023 OR PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026)',
}
results = {}
for k,q in ARXIV.items():
    url = 'http://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': q, 'start':0, 'max_results': 30, 'sortBy':'relevance'})
    try:
        root = ET.fromstring(get(url)); ns={'a':'http://www.w3.org/2005/Atom'}; rows=[]
        for e in root.findall('a:entry', ns):
            rows.append({'id': e.find('a:id',ns).text.strip(), 'title': ' '.join(e.find('a:title',ns).text.split()), 'published': e.find('a:published',ns).text[:10], 'summary': ' '.join(e.find('a:summary',ns).text.split())[:700]})
        results['arxiv:'+k]=rows; print('arxiv',k,len(rows),flush=True)
    except Exception as ex: results['arxiv:'+k]={'error':str(ex)}; print('arxiv',k,'ERR',ex,flush=True)
    time.sleep(3.2)
for k,q in EPMC.items():
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode({'query': q, 'format':'json','pageSize':40,'sort':'RELEVANCE','resultType':'core'})
    try:
        d=json.loads(get(url)); rows=[]
        for r in d.get('resultList',{}).get('result',[]):
            rows.append({'pmid': r.get('pmid'), 'title': r.get('title'), 'journal': r.get('journalTitle'), 'year': r.get('pubYear'), 'abstract': (r.get('abstractText') or '')[:700]})
        results['epmc:'+k]=rows; print('epmc',k,len(rows),flush=True)
    except Exception as ex: results['epmc:'+k]={'error':str(ex)}; print('epmc',k,'ERR',ex,flush=True)
    time.sleep(1)
(OUT/'scan_novelty.json').write_text(json.dumps(results,indent=1,ensure_ascii=False)); print('DONE')
