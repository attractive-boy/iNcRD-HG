"""Public metadata search; papers are research sources, never training inputs."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE.parent / 'midrug_literature_20261006/expression_followup_search.py'
spec = importlib.util.spec_from_file_location('public_literature_fetch', source)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
module.HERE = HERE
QUERIES = [
    ('europepmc', '(TITLE_ABS:miRNA OR TITLE_ABS:microRNA) AND TITLE_ABS:drug AND (TITLE_ABS:prediction OR TITLE_ABS:predicting) AND FIRST_PDATE:[2025-01-01 TO 2026-10-06]'),
    ('crossref', 'graph link prediction weight averaging auxiliary learning'),
    ('arxiv', '(all:"weight averaging" OR all:"exponential moving average") AND (all:"graph" OR all:"early stopping") AND submittedDate:[202501010000 TO 202610062359]'),
    ('arxiv', '(all:"auxiliary task" OR all:"auxiliary loss") AND (all:"fine-tuning" OR all:"generalization") AND submittedDate:[202501010000 TO 202610062359]'),
]

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(module.fetch, QUERIES))
    (HERE / 'search_results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
    for r in results:
        print(r['source'], r['status'], r.get('error', ''), len(r.get('papers', [])))
        for p in r.get('papers', []): print(p.get('date'), p.get('doi', p.get('id')), p['title'])
