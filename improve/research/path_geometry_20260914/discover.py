"""Bounded public-source discovery after the functional-transfer rejection."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import requests

OUT=Path(__file__).resolve().parent
JOBS=[
    ('arxiv_path','https://export.arxiv.org/api/query',dict(search_query='all:"link prediction" AND (all:"path reasoning" OR all:"neural symbolic")',start=0,max_results=8,sortBy='submittedDate',sortOrder='descending')),
    ('crossref_path','https://api.crossref.org/works',dict(query='link prediction path reasoning neural symbolic',filter='from-pub-date:2024-01-01,until-pub-date:2026-09-14',rows=8)),
    ('crossref_geometry','https://api.crossref.org/works',dict(query='hyperbolic heterogeneous link prediction',filter='from-pub-date:2025-01-01,until-pub-date:2026-09-14',rows=6)),
    ('openalex_path','https://api.openalex.org/works',dict(search='link prediction path reasoning',filter='from_publication_date:2025-01-01,to_publication_date:2026-09-14',per_page=8)),
    ('openalex_topology','https://api.openalex.org/works',dict(search='link prediction non backtracking topology',filter='from_publication_date:2025-01-01,to_publication_date:2026-09-14',per_page=6)),
]


def fetch(job):
    name,url,params=job
    record=dict(name=name,url=url,params=params,requested_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        r=requests.get(url,params=params,timeout=(8,25),headers={'User-Agent':'incrd-literature-audit/1.0'})
        suffix='.xml' if 'arxiv' in name else '.json'
        path=OUT/(name+suffix);path.write_bytes(r.content)
        record.update(status=r.status_code,final_url=r.url,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest())
        r.raise_for_status()
        if 'crossref' in name:
            record['hits']=[dict(title=(x.get('title') or [''])[0],doi=x.get('DOI'),date=x.get('published'),journal=x.get('container-title'),type=x.get('type'),abstract=x.get('abstract','')) for x in r.json()['message']['items']]
        elif 'openalex' in name:
            record['hits']=[dict(title=x.get('title'),doi=x.get('doi'),date=x.get('publication_date'),id=x.get('id'),type=x.get('type'),location=x.get('primary_location')) for x in r.json()['results']]
    except Exception as exc:record['error']=str(exc)
    return record


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,JOBS))
    (OUT/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    for row in records:
        print(row['name'],row.get('status'),row.get('error',''))
        for hit in row.get('hits',[]):print(hit['title'],hit.get('doi'),hit.get('date'))
