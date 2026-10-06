"""Round-12 frontier scan: cycles/motifs, masked graph training, pair relabeling, degree-corrected
models, whole-matrix objectives, and 2026 miRNA/ncRNA-drug association models."""
import json, ssl, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from pathlib import Path
OUT = Path(__file__).resolve().parent
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
def get(url, timeout=60, tries=4):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 LitScan/1.4 (mailto:lit@example.org)'})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r: return r.read()
        except Exception:
            if t == tries - 1: raise
            time.sleep(15 * (t + 1))
ARXIV = {
 'motif_cycle_lp': '(all:"motif" OR all:"cycle" OR all:"butterfly") AND (all:"link prediction")',
 'bipartite_lp_2025': '(all:"bipartite" OR all:"user-item") AND (all:"link prediction" OR all:"recommendation") AND (all:"graph neural network")',
 'degree_corrected': '(all:"degree-corrected" OR all:"configuration model" OR all:"debiased") AND (all:"link prediction" OR all:"collaborative filtering")',
 'subgraph_lp_2026': '(all:"subgraph" OR all:"labeling trick" OR all:"target-aware") AND (all:"link prediction")',
 'pair_encoder': '(all:"pair representation" OR all:"pairwise") AND (all:"graph neural network" OR all:"transformer") AND (all:"link prediction" OR all:"interaction")',
 'whole_matrix_obj': '(all:"full softmax" OR all:"sampled softmax" OR all:"non-sampling" OR all:"listwise") AND (all:"recommendation" OR all:"collaborative filtering" OR all:"retrieval")',
 'ssl_masked_edge': '(all:"masked edge" OR all:"edge masking" OR all:"masked autoencoder") AND (all:"link prediction" OR all:"bipartite" OR all:"recommendation")',
 'mda_models_2026': '(all:"miRNA" OR all:"ncRNA" OR all:"lncRNA") AND (all:"drug" OR all:"drug resistance") AND (all:"prediction" OR all:"association")',
 'self_distill_gnn': '(all:"self-distillation" OR all:"co-training" OR all:"teacher-student") AND (all:"graph neural network") AND (all:link OR all:recommendation OR all:prediction)',
 'structural_pe': '(all:"random walk" OR all:"spectral") AND (all:"positional encoding" OR all:"structural encoding") AND (all:"graph neural network") AND (all:"link prediction" OR all:"bipartite")',
 'popularity_shortcut': '(all:"shortcut" OR all:"spurious") AND (all:"graph neural network" OR all:"collaborative filtering") AND (all:"degree" OR all:"popularity")',
 'diffusion_lp': '(all:"diffusion" OR all:"heat kernel" OR all:"personalized PageRank") AND (all:"link prediction")',
}
EPMC = {
 'mda_2026': '("miRNA-drug" OR "microRNA-drug" OR "miRNA drug") AND (association OR resistance OR sensitivity) AND prediction AND (PUB_YEAR:2026)',
 'ncrna_assoc_2026': '("ncRNA" OR "miRNA" OR "lncRNA" OR "circRNA") AND ("drug" OR "disease") AND ("association prediction" OR "link prediction") AND (PUB_YEAR:2026)',
}
results = {}
for k, q in ARXIV.items():
    try:
        url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': q, 'start': 0, 'max_results': 30, 'sortBy': 'submittedDate', 'sortOrder': 'descending'})
        root = ET.fromstring(get(url)); ns = {'a': 'http://www.w3.org/2005/Atom'}
        items = [dict(id=e.find('a:id', ns).text.split('/abs/')[-1], date=e.find('a:published', ns).text[:10],
                      title=' '.join(e.find('a:title', ns).text.split()), abstract=' '.join((e.find('a:summary', ns).text or '').split())[:1200]) for e in root.findall('a:entry', ns)]
        results['arxiv:' + k] = items; print('arxiv', k, len(items), flush=True)
    except Exception as ex:
        results['arxiv:' + k] = dict(error=str(ex)); print('arxiv', k, 'ERR', ex, flush=True)
    time.sleep(6)
for k, q in EPMC.items():
    try:
        url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode({'query': q, 'format': 'json', 'pageSize': 50, 'resultType': 'core', 'sort': 'P_PDATE_D desc'})
        js = json.loads(get(url))
        items = [dict(id=r.get('pmid') or r.get('doi') or r.get('id'), date=r.get('firstPublicationDate', ''), title=r.get('title', ''), abstract=(r.get('abstractText') or '')[:1200]) for r in js.get('resultList', {}).get('result', [])]
        results['epmc:' + k] = items; print('epmc', k, len(items), flush=True)
    except Exception as ex:
        results['epmc:' + k] = dict(error=str(ex)); print('epmc', k, 'ERR', ex, flush=True)
    time.sleep(2)
json.dump(results, open(OUT / 'scan_r12.json', 'w'), indent=1)
with open(OUT / 'scan_r12_titles.txt', 'w') as f:
    for k, v in results.items():
        f.write(f'### {k}\n')
        for it in (v if isinstance(v, list) else []):
            f.write(f"{it['date']} {it['id']} {it['title']}\n")
print('done')
