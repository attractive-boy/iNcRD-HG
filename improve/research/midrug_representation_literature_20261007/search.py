"""Recent representation regularization: literature only, no training inputs."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE.parent / 'midrug_literature_20261006/expression_followup_search.py'
spec = importlib.util.spec_from_file_location('public_representation_fetch', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.HERE = HERE
QUERIES = [
    ('europepmc', '(TITLE_ABS:miRNA OR TITLE_ABS:microRNA) AND TITLE_ABS:drug AND (TITLE_ABS:contrastive OR TITLE_ABS:disentangled OR TITLE_ABS:redundancy) AND FIRST_PDATE:[2025-01-01 TO 2026-10-06]'),
    ('crossref', 'miRNA drug association prediction redundancy reduction contrastive learning'),
    ('arxiv', '(all:"link prediction") AND (all:"decorrelation" OR all:"redundancy reduction" OR all:"whitening" OR all:"representation collapse") AND submittedDate:[202401010000 TO 202610062359]'),
    ('arxiv', '(all:"graph") AND (all:"variance covariance" OR all:"VICReg" OR all:"Barlow Twins") AND submittedDate:[202501010000 TO 202610062359]'),
]

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(module.fetch, QUERIES))
    (HERE / 'search_results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
    merged = {}
    for r in results:
        print(r['source'], r['status'], r.get('error', ''), len(r.get('papers', [])), flush=True)
        for p in r.get('papers', []):
            key = p.get('doi') or p.get('id') or ' '.join(p['title'].lower().split())
            merged.setdefault(key, dict(p, sources=[]))['sources'].append(r['source'])
            print(p.get('date'), p.get('doi', p.get('id')), p['title'], flush=True)
    (HERE / 'deduplicated.json').write_text(json.dumps(list(merged.values()), indent=2, ensure_ascii=False))
