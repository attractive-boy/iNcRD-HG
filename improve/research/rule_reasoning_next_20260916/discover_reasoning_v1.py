"""Bounded public-source search for trainable relational reasoning mechanisms."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import requests

HERE = Path(__file__).resolve().parent
OUT = HERE/'REASONING_DISCOVERY_V1.json'
if OUT.exists():
    raise ValueError('Preserve search result')
JOBS = [
    ('arxiv_paths', 'https://export.arxiv.org/api/query',
     {'search_query':'(all:"Bellman-Ford" OR all:"A*Net" OR all:"neural rule learning") AND submittedDate:[202401010000 TO 202609162359]',
      'max_results':12,'sortBy':'submittedDate','sortOrder':'descending'}),
    ('arxiv_logic', 'https://export.arxiv.org/api/query',
     {'search_query':'(ti:"rule learning" OR ti:"logical reasoning") AND (all:"knowledge graph" OR all:"link prediction") AND submittedDate:[202401010000 TO 202609162359]',
      'max_results':10,'sortBy':'submittedDate','sortOrder':'descending'}),
    ('crossref_paths','https://api.crossref.org/works',
     {'query':'neural Bellman Ford knowledge graph reasoning link prediction','filter':'from-pub-date:2024-01-01','rows':6}),
]


def fetch(job):
    name,url,params = job
    row = dict(source=name,url=url,params=params,utc=datetime.now(timezone.utc).isoformat())
    try:
        response = requests.get(url,params=params,timeout=35)
        path = HERE/(name+'_v1.'+('xml' if name.startswith('arxiv') else 'json'))
        if path.exists():
            raise ValueError('Preserve source response')
        path.write_bytes(response.content)
        row.update(http_status=response.status_code,file=str(path),sha256=hashlib.sha256(response.content).hexdigest())
        response.raise_for_status()
        if name.startswith('arxiv'):
            ns = {'a':'http://www.w3.org/2005/Atom'}
            row['items'] = [dict(id=e.findtext('a:id',namespaces=ns),
                title=' '.join(e.findtext('a:title',namespaces=ns).split()),
                published=e.findtext('a:published',namespaces=ns),
                summary=' '.join(e.findtext('a:summary',namespaces=ns).split()))
                for e in ET.fromstring(response.content).findall('a:entry',ns)]
        else:
            row['items'] = [{key:item.get(key) for key in ('title','DOI','published','container-title')}
                            for item in response.json()['message']['items']]
    except Exception as exc:
        row['error'] = str(exc)
    return row


with ThreadPoolExecutor(max_workers=3) as pool:
    data = list(pool.map(fetch,JOBS))
OUT.write_text(json.dumps(data,indent=2)+'\n')
for item in data:
    print(item['source'],item.get('http_status'),item.get('error',''),flush=True)
    for result in item.get('items',[]):
        print(result.get('id',result.get('DOI')),result['title'],flush=True)
