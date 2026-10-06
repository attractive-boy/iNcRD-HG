"""Delta literature scan (angles not in plmf_frontier_20260922/lit/search.py)."""
import json, ssl, urllib.request, urllib.parse, xml.etree.ElementTree as ET, time
from pathlib import Path
OUT = Path(__file__).resolve().parent
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def get(url, timeout=40):
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0 LitScan/1.0'})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r: return r.read()
Q = {
 'multitask_lp': 'all:"link prediction" AND (all:"multi-task" OR all:"multi-relational" OR all:"auxiliary") AND all:bipartite',
 'cross_domain_bipartite': '(all:"cross-domain" OR all:"transfer learning") AND (all:bipartite OR all:"user-item") AND (all:"link prediction" OR all:"matrix completion")',
 'multi_behavior': 'all:"multi-behavior" AND all:recommendation AND all:2026',
 'side_info_mc': '(all:"matrix completion" OR all:"collaborative filtering") AND all:"side information" AND all:inductive',
 'label_noise_lp': '("label noise" OR "noisy labels" OR "false negatives") AND all:"link prediction"',
 'degree_corrected': '(all:"degree-corrected" OR all:"degree correction" OR all:"popularity debias") AND (all:"link prediction" OR all:recommendation)',
 'learnable_heuristic': '(all:"learnable heuristic" OR all:"heuristic learning" OR all:"polynomial filter") AND all:"link prediction"',
 'conformal_lp': 'all:conformal AND (all:"link prediction" OR all:"graph")',
 'drug_resistance_mirna': 'all:"drug resistance" AND (all:miRNA OR all:microRNA) AND (all:graph OR all:"deep learning")',
 'bio_association_transfer': '(all:"association prediction") AND (all:"transfer" OR all:"multi-task" OR all:"cross-dataset")',
 'mirna_drug_2026': 'all:miRNA AND all:drug AND all:association AND all:2026',
 'llm_embedding_bio_lp': '(all:"LLM" OR all:"language model") AND all:embedding AND (all:miRNA OR all:"drug-target") AND all:prediction',
}
res = {}
for k, q in Q.items():
    url = 'http://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': q, 'start': 0, 'max_results': 40, 'sortBy': 'submittedDate', 'sortOrder': 'descending'})
    try:
        x = get(url); root = ET.fromstring(x); ns = {'a': 'http://www.w3.org/2005/Atom'}
        items = []
        for e in root.findall('a:entry', ns):
            pub = e.find('a:published', ns).text[:10]
            if pub < '2025-06-01': continue
            items.append(dict(id=e.find('a:id', ns).text.split('/abs/')[-1], date=pub, title=' '.join(e.find('a:title', ns).text.split()), abs=' '.join(e.find('a:summary', ns).text.split())[:600]))
        res[k] = items; print(k, len(items), flush=True)
    except Exception as ex:
        print(k, 'ERR', ex, flush=True)
    time.sleep(3)
json.dump(res, open(OUT / 'delta_results.json', 'w'), indent=1, ensure_ascii=False)
for k, v in res.items():
    print('\n##', k)
    for it in v[:25]: print(f"  {it['date']} {it['id']} {it['title']}")
