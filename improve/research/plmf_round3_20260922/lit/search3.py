"""Round-3 frontier scan: angles NOT covered by plmf_frontier_20260922/lit/search.py."""
import json, ssl, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from pathlib import Path
OUT = Path(__file__).resolve().parent
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def get(url, timeout=40):
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0 LitScan/3.0 (mailto:lit@example.org)'})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r: return r.read()
ARXIV = {
 'cold_pop_impute': 'all:"cold-start" AND (all:"popularity" OR all:"degree") AND (all:recommendation OR all:"link prediction") AND (all:"content" OR all:"side information")',
 'ap_loss': '(all:"average precision" OR all:AUPRC OR all:"precision-recall") AND (all:"surrogate loss" OR all:"direct optimization" OR all:"differentiable") AND (all:imbalance OR all:"link prediction" OR all:retrieval)',
 'pu_bio': '(all:"positive-unlabeled" OR all:"PU learning") AND (all:drug OR all:miRNA OR all:"association prediction" OR all:"biomedical")',
 'self_training_lp': '(all:"self-training" OR all:"pseudo-label" OR all:"test-time") AND all:"link prediction"',
 'llm_kg_bio_lp': '(all:"large language model" OR all:LLM) AND (all:"drug" OR all:"biomedical") AND (all:"link prediction" OR all:"association prediction" OR all:"knowledge graph completion") AND all:2026',
 'noisy_neg_lp': '(all:"false negative" OR all:"negative sampling" OR all:"label noise") AND all:"link prediction" AND (all:biomedical OR all:"knowledge graph" OR all:drug)',
 'hard_neg_bio': '(all:"hard negative" OR all:"degree-matched" OR all:"debiased") AND (all:"drug-target" OR all:"drug" OR all:"miRNA") AND all:prediction',
 'few_shot_lp': '(all:"few-shot" OR all:"meta-learning") AND (all:"link prediction" OR all:"cold-start") AND (all:"drug" OR all:"recommendation" OR all:"bipartite")',
 'ot_lp': '(all:"optimal transport" OR all:"Sinkhorn") AND (all:"link prediction" OR all:"bipartite" OR all:"drug-target")',
 'conformal_bio_lp': '(all:conformal OR all:"calibration") AND (all:"link prediction" OR all:"drug-target" OR all:"association prediction")',
 'mda_2026': '(all:miRNA OR all:microRNA) AND (all:drug) AND (all:resistance OR all:sensitivity OR all:response) AND (all:graph OR all:"deep learning")',
 'foundation_bio_lp': '(all:"foundation model" OR all:"pretrained") AND (all:RNA OR all:molecule) AND (all:"association" OR all:"interaction") AND all:"cold"',
 'graph_lp_2026_new': 'all:"link prediction" AND (all:"generative" OR all:"diffusion" OR all:"flow matching") AND all:graph',
 'sheaf_hyper_lp': '(all:sheaf OR all:hypergraph OR all:"simplicial") AND all:"link prediction" AND (all:bipartite OR all:recommendation OR all:biomedical)',
}
EPMC = {
 'mda_2026_late': '(miRNA OR microRNA OR ncRNA OR lncRNA OR circRNA) AND drug AND (resistance OR sensitivity) AND (association OR interaction) AND prediction AND (FIRST_PDATE:[2026-05-01 TO 2026-12-31])',
 'cold_bio_2026': '("cold-start" OR "cold start" OR inductive OR "unseen drugs" OR "novel drugs") AND (association OR interaction) AND prediction AND (miRNA OR lncRNA OR circRNA OR "drug-target") AND (FIRST_PDATE:[2026-01-01 TO 2026-12-31])',
 'pu_bio_2026': '("positive-unlabeled" OR "positive unlabeled") AND (drug OR miRNA OR lncRNA) AND prediction AND (PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'llm_ncrna_drug': '("large language model" OR LLM OR GPT OR "text embedding") AND (miRNA OR ncRNA OR lncRNA OR circRNA) AND drug AND prediction AND (PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'degree_bias_bio_2026': '("degree bias" OR "popularity bias" OR "hubness" OR "node degree" OR "shortcut learning") AND (association prediction OR "link prediction") AND (drug OR miRNA OR biomedical) AND (PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'aupr_loss_bio': '("average precision" OR AUPR OR AUPRC) AND ("loss function" OR "ranking loss" OR "direct optimization") AND (drug OR association prediction) AND (PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'plmf_cites2': '"PLMF-MDA" OR ("pretrained language model" AND miRNA AND drug AND resistance)',
}
results = {}
for k,q in ARXIV.items():
    url = 'http://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': q, 'start':0, 'max_results': 40, 'sortBy':'submittedDate', 'sortOrder':'descending'})
    try:
        root = ET.fromstring(get(url)); ns={'a':'http://www.w3.org/2005/Atom'}; rows=[]
        for e in root.findall('a:entry', ns):
            rows.append({'id': e.find('a:id',ns).text.strip(), 'title': ' '.join(e.find('a:title',ns).text.split()),
                         'published': e.find('a:published',ns).text[:10], 'summary': ' '.join(e.find('a:summary',ns).text.split())[:900]})
        results['arxiv:'+k]=rows; print('arxiv',k,len(rows),flush=True)
    except Exception as ex:
        results['arxiv:'+k]={'error':str(ex)}; print('arxiv',k,'ERR',ex,flush=True)
    time.sleep(3.2)
for k,q in EPMC.items():
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode({'query': q, 'format':'json','pageSize':50,'sort':'P_PDATE_D desc','resultType':'lite'})
    try:
        d=json.loads(get(url)); rows=[]
        for r in d.get('resultList',{}).get('result',[]):
            rows.append({'id': r.get('doi') or r.get('id'), 'title': r.get('title'), 'journal': r.get('journalTitle'), 'published': r.get('firstPublicationDate'), 'pmid': r.get('pmid')})
        results['epmc:'+k]=rows; print('epmc',k,len(rows),flush=True)
    except Exception as ex:
        results['epmc:'+k]={'error':str(ex)}; print('epmc',k,'ERR',ex,flush=True)
    time.sleep(1.5)
json.dump(results, open(OUT/'scan3_results.json','w'), indent=1, ensure_ascii=False)
