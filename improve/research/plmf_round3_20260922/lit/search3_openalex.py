import json, ssl, time, urllib.request, urllib.parse
from pathlib import Path
OUT = Path(__file__).resolve().parent
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def get(url, timeout=60):
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0 LitScan/3.0 (mailto:lit@example.org)'})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r: return r.read()
Q = {
 'cold_pop_impute': 'cold-start item popularity prediction content side information recommendation',
 'inherited_popularity_cold': 'inherited popularity bias cold-start item recommendation',
 'ap_loss': 'average precision surrogate loss direct optimization AUPRC',
 'pu_bio': 'positive-unlabeled learning drug association prediction',
 'self_training_lp': 'self-training pseudo-label link prediction graph',
 'llm_kg_bio_lp': 'large language model biomedical knowledge graph link prediction drug',
 'noisy_neg_lp': 'false negative negative sampling biomedical link prediction',
 'few_shot_lp': 'few-shot link prediction cold-start drug bipartite meta-learning',
 'ot_lp': 'optimal transport bipartite link prediction drug-target',
 'conformal_lp': 'conformal prediction link prediction drug-target',
 'mda_2026': 'miRNA drug resistance association prediction graph neural network',
 'generative_lp': 'diffusion model link prediction graph generative',
 'degree_prediction': 'predicting node degree from attributes unseen nodes inductive link prediction popularity',
 'literature_prior_drug': 'literature count bias drug prediction benchmark publication bias',
 'hubness_bio_lp': 'hub bias degree correction biomedical link prediction evaluation',
}
res = {}
for k, q in Q.items():
    url = 'https://api.openalex.org/works?' + urllib.parse.urlencode({'search': q, 'per-page': 25, 'filter': 'from_publication_date:2025-01-01', 'select': 'id,doi,title,publication_date,primary_location,cited_by_count,abstract_inverted_index'})
    try:
        d = json.loads(get(url)); rows = []
        for w in d.get('results', []):
            inv = w.get('abstract_inverted_index') or {}
            words = sorted(((p, t) for t, ps in inv.items() for p in ps)); abs_ = ' '.join(t for _, t in words)[:900]
            src = ((w.get('primary_location') or {}).get('source') or {}).get('display_name')
            rows.append({'id': w.get('doi') or w.get('id'), 'title': w.get('title'), 'published': w.get('publication_date'), 'source': src, 'cited': w.get('cited_by_count'), 'summary': abs_})
        res[k] = rows; print(k, len(rows), flush=True)
    except Exception as ex:
        res[k] = {'error': str(ex)}; print(k, 'ERR', ex, flush=True)
    time.sleep(2)
json.dump(res, open(OUT/'scan3_openalex_rel.json', 'w'), indent=1, ensure_ascii=False)
