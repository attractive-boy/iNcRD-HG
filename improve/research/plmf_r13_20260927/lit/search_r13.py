"""Round-13 frontier scan: topics not covered by rounds 1-12 (see ../../plmf_*/lit/*titles*.txt):
test-time adaptation, exchangeable/equivariant matrix models, inductive (subgraph) matrix completion,
sharpness/weight averaging, KAN decoders, influence / leave-one-out, tail-node LP, residual/label propagation,
edge augmentation, counterfactual LP, graph structure learning, 2026 MDA models."""
import json, ssl, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from pathlib import Path
OUT = Path(__file__).resolve().parent
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
def get(url, timeout=60, tries=4):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 LitScan/1.5 (mailto:lit@example.org)'})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r: return r.read()
        except Exception:
            if t == tries - 1: raise
            time.sleep(15 * (t + 1))
ARXIV = {
 'tta_lp': '(all:"test-time" OR all:"test time training" OR all:"test-time adaptation") AND (all:"link prediction" OR all:"recommendation" OR all:"graph neural network")',
 'exchangeable_matrix': '(all:"exchangeable" OR all:"permutation equivariant" OR all:"equivariant") AND (all:"matrix completion" OR all:"interaction" OR all:"bipartite")',
 'inductive_mc_subgraph': '(all:"inductive matrix completion" OR all:"IGMC" OR all:"enclosing subgraph") AND (all:"graph" OR all:"link")',
 'sam_swa_gnn': '(all:"sharpness-aware" OR all:"weight averaging" OR all:"model soup" OR all:"stochastic weight averaging") AND (all:"graph neural network" OR all:"link prediction" OR all:"recommendation")',
 'kan_lp': '(all:"Kolmogorov-Arnold" OR all:"KAN") AND (all:"link prediction" OR all:"association prediction" OR all:"graph neural network")',
 'influence_loo_lp': '(all:"influence function" OR all:"leave-one-out" OR all:"jackknife" OR all:"unlearning") AND (all:"link prediction" OR all:"collaborative filtering" OR all:"graph neural network")',
 'tail_node_lp': '(all:"tail node" OR all:"long-tail" OR all:"low-degree" OR all:"cold-start") AND (all:"link prediction") AND (all:"graph")',
 'residual_prop': '(all:"correct and smooth" OR all:"label propagation" OR all:"residual propagation" OR all:"error propagation") AND (all:"link prediction" OR all:"graph neural network")',
 'edge_augment_lp': '(all:"data augmentation" OR all:"edge augmentation" OR all:"mixup") AND (all:"link prediction")',
 'counterfactual_lp': '(all:"counterfactual" OR all:"causal") AND (all:"link prediction")',
 'gsl_bipartite': '(all:"graph structure learning" OR all:"graph rewiring") AND (all:"link prediction" OR all:"recommendation" OR all:"bipartite")',
 'gp_kernel_lp': '(all:"Gaussian process" OR all:"kernel ridge" OR all:"neural tangent kernel") AND (all:"link prediction" OR all:"matrix completion" OR all:"drug")',
 'bayes_mc_lp': '(all:"Bayesian" OR all:"uncertainty") AND (all:"link prediction" OR all:"matrix completion") AND (all:"sparse" OR all:"cold")',
 'lp_2026_latest': '(all:"link prediction") AND (all:"graph neural network" OR all:"graph transformer")',
}
EPMC = {
 'mda_2026b': '("miRNA" OR "microRNA") AND ("drug" OR "drug resistance" OR "drug sensitivity") AND ("prediction" OR "association") AND ("graph" OR "deep learning") AND (PUB_YEAR:2026)',
 'assoc_pred_novel_2026': '("association prediction") AND ("graph neural network" OR "hypergraph" OR "transformer") AND (PUB_YEAR:2026)',
}
results = {}
for k, q in ARXIV.items():
    try:
        url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': q, 'start': 0, 'max_results': 30, 'sortBy': 'submittedDate', 'sortOrder': 'descending'})
        root = ET.fromstring(get(url)); ns = {'a': 'http://www.w3.org/2005/Atom'}
        items = [dict(id=e.find('a:id', ns).text.split('/abs/')[-1], date=e.find('a:published', ns).text[:10],
                      title=' '.join(e.find('a:title', ns).text.split()), abstract=' '.join((e.find('a:summary', ns).text or '').split())[:1500]) for e in root.findall('a:entry', ns)]
        results['arxiv:' + k] = items; print('arxiv', k, len(items), flush=True)
    except Exception as ex:
        results['arxiv:' + k] = dict(error=str(ex)); print('arxiv', k, 'ERR', ex, flush=True)
    time.sleep(6)
for k, q in EPMC.items():
    try:
        url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode({'query': q, 'format': 'json', 'pageSize': 60, 'resultType': 'core', 'sort': 'P_PDATE_D desc'})
        js = json.loads(get(url))
        items = [dict(id=r.get('pmid') or r.get('doi') or r.get('id'), date=r.get('firstPublicationDate', ''), title=r.get('title', ''), abstract=(r.get('abstractText') or '')[:1500]) for r in js.get('resultList', {}).get('result', [])]
        results['epmc:' + k] = items; print('epmc', k, len(items), flush=True)
    except Exception as ex:
        results['epmc:' + k] = dict(error=str(ex)); print('epmc', k, 'ERR', ex, flush=True)
    time.sleep(2)
json.dump(results, open(OUT / 'scan_r13.json', 'w'), indent=1)
with open(OUT / 'scan_r13_titles.txt', 'w') as f:
    for k, v in results.items():
        f.write(f'### {k}\n')
        for it in (v if isinstance(v, list) else []):
            f.write(f"{it['date']} {it['id']} {it['title']}\n")
print('done')
