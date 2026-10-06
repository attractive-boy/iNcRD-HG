"""Delta literature scan focused on WARM single-model gains for bipartite association prediction (2025-2026)."""
import json, ssl, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET, re
from pathlib import Path
OUT = Path(__file__).resolve().parent
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def get(url, timeout=40):
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0 LitScan/1.0 (mailto:lit@example.org)'})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return r.read()
ARXIV = {
 'pu_lp': '(all:"positive-unlabeled" OR all:"PU learning" OR all:"hidden positives" OR all:"false negatives") AND (all:"link prediction" OR all:"collaborative filtering" OR all:"association prediction")',
 'label_noise_lp': '(all:"label noise" OR all:"noisy labels" OR all:"missing not at random") AND (all:"link prediction" OR all:"implicit feedback" OR all:"association prediction")',
 'auc_opt': '(all:"AUC maximization" OR all:"AUPRC" OR all:"average precision" OR all:"ranking loss") AND (all:"link prediction" OR all:"recommendation" OR all:"association prediction") AND all:2025',
 'lp_sota_2026': 'all:"link prediction" AND (all:"state-of-the-art" OR all:"outperform") AND (all:"graph neural" OR all:"transformer") AND all:2026',
 'linear_ae': '(all:"linear autoencoder" OR all:EASE OR all:"closed-form") AND (all:"collaborative filtering" OR all:recommendation)',
 'spectral_cf_2026': '(all:"spectral" OR all:"graph filter" OR all:"polynomial filter") AND (all:"collaborative filtering" OR all:recommendation) AND all:2026',
 'hypergraph_cf': 'all:hypergraph AND (all:"collaborative filtering" OR all:"link prediction" OR all:"association prediction") AND all:2026',
 'self_training_lp': '(all:"self-training" OR all:"pseudo-label" OR all:"iterative") AND (all:"link prediction" OR all:"matrix completion" OR all:"graph completion")',
 'negative_sampling_lp': '(all:"negative sampling" OR all:"hard negative") AND (all:"link prediction" OR all:"collaborative filtering") AND (all:2025 OR all:2026)',
 'mixture_lp': '(all:"mixture of experts" OR all:"adaptive") AND all:"link prediction" AND all:2025',
 'diffusion_lp_2026': '(all:"diffusion model" OR all:"flow matching" OR all:"generative") AND (all:"link prediction" OR all:"collaborative filtering") AND all:2026',
 'llm_association_bio': '(all:"large language model" OR all:LLM OR all:"language model") AND (all:miRNA OR all:"drug-target" OR all:"drug response") AND all:association',
 'mirna_drug_2026': '(all:miRNA OR all:microRNA) AND all:drug AND (all:resistance OR all:sensitivity OR all:association) AND all:2026',
 'kge_side_info_2026': '(all:"matrix completion" OR all:"knowledge graph embedding") AND (all:"side information" OR all:"inductive") AND all:2026',
 'gnn_lp_theory_2026': '(all:"link prediction") AND (all:"expressive" OR all:"heuristic" OR all:"common neighbor") AND all:2026',
 'calibration_lp': '(all:calibration OR all:"conformal") AND all:"link prediction"',
 'text_mining_lp': '(all:"literature" OR all:"co-occurrence" OR all:"text mining") AND (all:"association prediction" OR all:"link prediction") AND all:biomedical',
}
EPMC = {
 'mirna_drug_2026': '((miRNA OR microRNA) AND drug AND (resistance OR sensitivity) AND (association OR interaction) AND (prediction OR predicting)) AND (PUB_YEAR:2026)',
 'ncrna_drug_2026': '(("non-coding RNA" OR ncRNA OR lncRNA OR circRNA) AND drug AND (resistance OR sensitivity OR response) AND association AND (graph OR "deep learning" OR "language model")) AND (PUB_YEAR:2026)',
 'plmf_mentions': '"PLMF-MDA" OR ("miRNA-drug resistance" AND "sensitivity associations" AND prediction)',
 'pu_bio_assoc': '("positive-unlabeled" OR "positive unlabeled") AND (miRNA OR lncRNA OR "drug-target" OR "drug-disease") AND association AND (PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'negatives_bio_assoc': '("negative samples" OR "negative sampling" OR "reliable negatives") AND (miRNA OR lncRNA OR circRNA) AND (association prediction) AND (PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'hypergraph_bio_assoc': '(hypergraph) AND (miRNA OR lncRNA OR circRNA OR drug) AND (association prediction) AND (PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'llm_bio_assoc_2026': '("large language model" OR LLM OR "language model") AND (miRNA OR ncRNA OR lncRNA) AND (drug OR disease) AND (association prediction) AND (PUB_YEAR:2026)',
 'benchmark_leak_bio': '(("data leakage" OR "degree bias" OR "shortcut") AND (association prediction OR "link prediction") AND (miRNA OR drug OR biomedical)) AND (PUB_YEAR:2025 OR PUB_YEAR:2026)',
 'mda_sota_2026': '((miRNA OR microRNA) AND (drug) AND (association) AND (AUC) AND (graph OR attention OR transformer OR contrastive)) AND (PUB_YEAR:2026)',
}
results = {}
for k,q in ARXIV.items():
    url = 'http://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': q, 'start':0, 'max_results': 40, 'sortBy':'submittedDate', 'sortOrder':'descending'})
    try:
        raw = get(url); root = ET.fromstring(raw); ns={'a':'http://www.w3.org/2005/Atom'}
        rows=[]
        for e in root.findall('a:entry', ns):
            rows.append({'id': e.find('a:id',ns).text.strip(), 'title': ' '.join(e.find('a:title',ns).text.split()),
                         'published': e.find('a:published',ns).text[:10], 'summary': ' '.join(e.find('a:summary',ns).text.split())[:1200]})
        results['arxiv:'+k]=rows; print('arxiv',k,len(rows),flush=True)
    except Exception as ex:
        results['arxiv:'+k]={'error':str(ex)}; print('arxiv',k,'ERR',ex,flush=True)
    time.sleep(3.2)
for k,q in EPMC.items():
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode({'query': q, 'format':'json','pageSize':60,'sort':'P_PDATE_D desc','resultType':'core'})
    try:
        d=json.loads(get(url)); rows=[]
        for r in d.get('resultList',{}).get('result',[]):
            rows.append({'pmid': r.get('pmid'), 'doi': r.get('doi'), 'title': r.get('title'), 'journal': r.get('journalTitle'), 'year': r.get('pubYear'), 'date': r.get('firstPublicationDate'), 'abstract': (r.get('abstractText') or '')[:1500]})
        results['epmc:'+k]=rows; print('epmc',k,len(rows),flush=True)
    except Exception as ex:
        results['epmc:'+k]={'error':str(ex)}; print('epmc',k,'ERR',ex,flush=True)
    time.sleep(1)
(OUT/'scan_warm.json').write_text(json.dumps(results,indent=1,ensure_ascii=False))
print('DONE')
