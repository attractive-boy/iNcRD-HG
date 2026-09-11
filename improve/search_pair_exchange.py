"""Targeted public-API prior-art check for conditional pair-exchange learning."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from improve.search_expression_consensus import crossref, arxiv, pubmed


QUERIES = (
    (crossref, 'An Econometric Model of Network Formation With Degree Heterogeneity'),
    (crossref, 'BPR Bayesian personalized ranking from implicit feedback'),
    (crossref, 'Neural Collaborative Filtering'),
    (crossref, 'Disentangling User Interest and Conformity for Recommendation with Causal Embedding'),
    (crossref, 'tetrad conditional likelihood bipartite network prediction'),
    (crossref, 'ncRNA drug resistance popularity bias conditional contrastive learning'),
    (arxiv, 'all:"degree-preserving" AND (all:"link prediction" OR all:"contrastive")'),
    (arxiv, 'all:"tetrad" AND (all:"network" OR all:"logit")'),
    (pubmed, '(ncRNA OR miRNA OR lncRNA) AND "drug resistance" AND (degree OR popularity OR bias OR "negative sampling" OR "matrix factorization")'),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    responses = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(fn, query): (fn.__name__, query) for fn, query in QUERIES}
        for future in as_completed(pending):
            source, query = pending[future]
            try:
                result = future.result()
            except Exception as error:
                result = dict(source=source, query=query, error=str(error), records=[])
            responses.append(result)
            print(source, len(result['records']), result.get('error', ''), query, flush=True)
    merged = {}
    for response in responses:
        for record in response['records']:
            key = (record.get('doi') or re.sub(r'\W+', '', record['title'])).lower()
            if key not in merged or len(record.get('abstract', '')) > len(merged[key].get('abstract', '')):
                merged[key] = record
    payload = dict(retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
                   scope='Targeted metadata/abstract search; not exhaustive novelty verification',
                   mcp_available=False, fallback='Public CrossRef, PubMed and arXiv APIs',
                   responses=responses, deduplicated=list(merged.values()))
    (out / 'search_results.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    for record in merged.values():
        print(record['source'], record.get('year'), record['title'], record.get('doi') or record['url'], flush=True)


if __name__ == '__main__':
    main()
