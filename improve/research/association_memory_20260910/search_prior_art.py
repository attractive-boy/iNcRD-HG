"""Targeted discovery for association-memory and conditional-parameter models."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from improve.search_expression_consensus import arxiv, crossref, pubmed


QUERIES = [
    (arxiv, 'all:"in-context" AND all:"link prediction"'),
    (arxiv, '(all:"retrieval" OR all:"case-based") AND all:"link prediction"'),
    (arxiv, 'all:"hypernetwork" AND (all:"drug" OR all:"molecular")'),
    (crossref, 'case based reasoning knowledge graph completion'),
    (crossref, 'drug target interaction retrieval memory neural process'),
    (crossref, 'hypernetwork drug response prediction'),
    (pubmed, '(ncRNA OR miRNA OR lncRNA) AND "drug resistance" AND (retrieval OR hypernetwork OR "neural process" OR "in-context")'),
    (pubmed, 'drug AND (hypernetwork OR "neural processes" OR "case-based reasoning")'),
]


def same(a, b):
    doi_a, doi_b = [(r.get('doi','') or '').lower().replace('https://doi.org/','') for r in (a,b)]
    if doi_a and doi_b:
        return doi_a == doi_b
    stop = {'a','an','the','in','of','for','on','to','and','with','by','et','al'}
    ta,tb = [set(re.findall(r'\w+',r['title'].lower()))-stop for r in (a,b)]
    aa,ab = [(r.get('authors') or [''])[0].lower().split()[-1:] for r in (a,b)]
    return bool(aa and ab and aa==ab and ta|tb) and len(ta&tb)/len(ta|tb)>=.9


def quality(r):
    return (bool(r.get('doi') and r.get('venue')), r.get('source')!='arXiv', bool(r.get('abstract')))


def main():
    out=Path(__file__).parent
    responses=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending={pool.submit(fn,q):(fn.__name__,q) for fn,q in QUERIES}
        for future in as_completed(pending):
            source,query=pending[future]
            try:
                response=future.result()
            except Exception as error:
                response=dict(source=source,query=query,error=str(error),records=[])
            responses.append(response)
            print(source,len(response['records']),response.get('error',''),query,flush=True)
    merged=[]
    for response in responses:
        for record in response['records']:
            duplicate=next((i for i,r in enumerate(merged) if same(r,record)),None)
            if duplicate is None:
                merged.append(record)
            elif quality(record)>quality(merged[duplicate]):
                merged[duplicate]=record
    payload=dict(retrieved_at_utc=datetime.now(timezone.utc).isoformat(),mcp_available=False,
                 scope='Targeted public-API discovery; not exhaustive novelty verification.',
                 responses=responses,deduplicated=merged)
    path=out/'search_results.json'
    if path.exists():
        raise ValueError('Preserve previous search results')
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    for r in merged:
        print(r['source'],r.get('year'),r['title'],r.get('doi') or r['url'],flush=True)


if __name__=='__main__':
    main()
