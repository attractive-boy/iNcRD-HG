"""Targeted public metadata/abstract review, not a novelty certificate."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

from improve.search_expression_consensus import arxiv, crossref, pubmed


JOBS = (
    (arxiv, 'ti:"Neural" AND ti:"Common" AND ti:"Neighbor"'),
    (arxiv, 'all:"link prediction" AND all:"higher-order"'),
    (arxiv, 'all:"bipartite" AND all:"multiplicative"'),
    (crossref, 'TankBind Trigonometry Aware Neural Networks Drug Protein Binding'),
    (crossref, 'Highly accurate protein structure prediction with AlphaFold'),
    (pubmed, '("drug resistance"[Title/Abstract]) AND ("association prediction"[Title/Abstract]) AND ("higher-order"[Title/Abstract] OR "matrix completion"[Title/Abstract] OR "diffusion"[Title/Abstract])'),
)


def main():
    out = Path(__file__).with_name('search_results.json')
    if out.exists():
        raise ValueError('Preserve the previous search')
    responses = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(fn, q): (fn.__name__, q) for fn, q in JOBS}
        for future in as_completed(pending):
            source, query = pending[future]
            try:
                r = future.result()
            except Exception as error:
                r = dict(source=source, query=query, error=str(error), records=[])
            responses.append(r)
            print(source, query, r.get('error', ''), flush=True)
            for record in r['records']:
                print(record.get('year'), record['title'], record.get('doi') or record['url'], flush=True)
    out.write_text(json.dumps(dict(retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Targeted metadata and abstract review only; no systematic novelty claim',
        mcp_available=False, fallback='Public arXiv, CrossRef, PubMed APIs',
        responses=responses), ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
