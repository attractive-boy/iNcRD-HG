"""Frontier literature scan (2025-2026) via reachable APIs: arXiv, Europe PMC, Crossref, bioRxiv."""
import json, ssl, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET, re, sys
from pathlib import Path
OUT = Path(__file__).resolve().parent
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def get(url, timeout=40):
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0 LitScan/1.0 (mailto:lit@example.org)'})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return r.read()
ARXIV = {
 'lp_heuristic': 'all:"link prediction" AND (all:"common neighbor" OR all:"heuristic learning" OR all:"pairwise encoding")',
 'lp_bipartite': 'all:"link prediction" AND all:bipartite',
 'lp_foundation': '(all:"graph foundation model" OR all:"in-context") AND all:"link prediction"',
 'gf_cf': '(all:"graph filtering" OR all:"graph signal processing") AND (all:"collaborative filtering" OR all:recommendation)',
 'diff_cf': '(all:"diffusion model" OR all:"flow matching") AND all:"collaborative filtering"',
 'coldstart_cf': 'all:"cold-start" AND (all:recommendation OR all:"collaborative filtering") AND (all:"content" OR all:"zero-shot")',
 'popbias': 'all:"popularity bias" AND (all:recommendation OR all:"link prediction")',
 'coldstart_dti': 'all:"cold-start" AND (all:"drug-target" OR all:"drug target interaction" OR all:"drug response")',
 'pu_link': '("positive-unlabeled" OR "PU learning") AND (all:"link prediction" OR all:"knowledge graph" OR all:biomedical)',
 'gt_link': '(all:"graph transformer" OR all:"positional encoding") AND all:"link prediction" AND all:2025',
 'spectral_lp': '(all:spectral OR all:"polynomial filter" OR all:"laplacian") AND all:"link prediction"',
 'llm_lp': '(all:"large language model" OR all:LLM) AND all:"link prediction" AND all:biomedical',
 'mirna_drug': 'all:miRNA AND all:drug AND (all:resistance OR all:sensitivity) AND all:prediction',
 'ncrna_drug': '(all:"non-coding RNA" OR all:ncRNA OR all:lncRNA OR all:circRNA) AND all:drug AND all:association AND all:prediction',
 'kg_inductive': 'all:"inductive" AND all:"knowledge graph completion" AND all:"unseen entities"',
 'lp_benchmark': 'all:"link prediction" AND (all:"evaluation" OR all:"benchmark") AND (all:"shortcut" OR all:"degree bias" OR all:"data leakage")',
 'hypergraph_lp': 'all:hypergraph AND all:"link prediction" AND all:2025',
 'ncn_mplp': 'all:"Neural Common Neighbor" OR all:"MPLP" OR all:"pure message passing" AND all:"link prediction"',
}
EPMC = {
 'mirna_drug_2026': '(miRNA OR microRNA) AND drug AND (resistance OR sensitivity) AND (association OR interaction) AND (prediction OR predicting) AND (PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'ncrna_drug_2026': '("non-coding RNA" OR ncRNA OR lncRNA OR circRNA) AND drug AND (resistance OR sensitivity) AND association AND (graph OR "deep learning") AND (PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'coldstart_bio': '"cold-start" AND (drug OR miRNA OR "drug-target") AND (prediction) AND (PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'plmf_cites': 'CITES:(41045000_MED) OR "PLMF-MDA"',
 'mphgnn_cites': '"MPHGNN" OR "metapath-guided heterogeneous graph neural network"',
 'mda_benchmark_bias': '(miRNA OR ncRNA OR lncRNA) AND (association prediction) AND ("degree bias" OR "popularity" OR "shortcut" OR "data leakage" OR "evaluation bias") AND (PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'graph_filter_bio': '("graph filtering" OR "graph signal") AND (association prediction) AND (miRNA OR lncRNA OR drug) AND (PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'diffusion_mda': '("diffusion model" OR "denoising diffusion") AND (miRNA OR lncRNA OR circRNA) AND (association prediction) AND (PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'llm_mda': '("large language model" OR LLM) AND (miRNA OR ncRNA) AND (drug OR disease) AND (association prediction) AND (PUB_YEAR:2025 OR PUB_YEAR:2026)',
}
results = {}
# arXiv
for k,q in ARXIV.items():
    url = 'http://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': q, 'start':0, 'max_results': 40, 'sortBy':'submittedDate', 'sortOrder':'descending'})
    try:
        raw = get(url); root = ET.fromstring(raw); ns={'a':'http://www.w3.org/2005/Atom'}
        rows=[]
        for e in root.findall('a:entry', ns):
            rows.append({'id': e.find('a:id',ns).text.strip(), 'title': ' '.join(e.find('a:title',ns).text.split()),
                         'published': e.find('a:published',ns).text[:10], 'summary': ' '.join(e.find('a:summary',ns).text.split())[:900]})
        results['arxiv:'+k]=rows; print('arxiv',k,len(rows),flush=True)
    except Exception as ex:
        results['arxiv:'+k]={'error':str(ex)}; print('arxiv',k,'ERR',ex,flush=True)
    time.sleep(3.2)
# Europe PMC
for k,q in EPMC.items():
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode({'query': q, 'format':'json','pageSize':50,'sort':'P_PDATE_D desc','resultType':'lite'})
    try:
        d=json.loads(get(url)); rows=[]
        for r in d.get('resultList',{}).get('result',[]):
            rows.append({'pmid': r.get('pmid'), 'doi': r.get('doi'), 'title': r.get('title'), 'journal': r.get('journalTitle'), 'year': r.get('pubYear'), 'date': r.get('firstPublicationDate'), 'pmcid': r.get('pmcid')})
        results['epmc:'+k]=rows; print('epmc',k,len(rows),flush=True)
    except Exception as ex:
        results['epmc:'+k]={'error':str(ex)}; print('epmc',k,'ERR',ex,flush=True)
    time.sleep(1)
# bioRxiv recent (category bioinformatics) - keyword filter
try:
    rows=[]
    for cursor in range(0, 600, 100):
        d=json.loads(get(f'https://api.biorxiv.org/details/biorxiv/2026-06-01/2026-09-22/{cursor}'))
        for r in d.get('collection',[]):
            t=(r.get('title','')+' '+r.get('abstract','')).lower()
            if re.search(r'mirna|microrna|ncrna|lncrna|circrna|drug resistance|link prediction|association prediction', t) and re.search(r'drug|resistance|link prediction', t):
                rows.append({'doi':r.get('doi'),'title':r.get('title'),'date':r.get('date'),'category':r.get('category'),'abstract':r.get('abstract','')[:600]})
        time.sleep(0.5)
    results['biorxiv:recent']=rows; print('biorxiv',len(rows),flush=True)
except Exception as ex:
    results['biorxiv:recent']={'error':str(ex)}; print('biorxiv ERR',ex)
(OUT/'scan_results.json').write_text(json.dumps(results,indent=1,ensure_ascii=False))
print('DONE')
