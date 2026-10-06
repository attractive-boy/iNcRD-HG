"""Bounded public literature discovery for measured perturbation morphology."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / 'MORPHOLOGY_DISCOVERY_V1.json'
if OUT.exists():
    raise ValueError('Preserve discovery record')

JOBS = [
    ('crossref_representation', 'https://api.crossref.org/works',
     {'query': 'Cell Painting morphological compound genetic perturbation representation learning',
      'filter': 'from-pub-date:2024-01-01,until-pub-date:2026-09-16', 'rows': 7}),
    ('crossref_jump', 'https://api.crossref.org/works',
     {'query': 'JUMP Cell Painting compound gene perturbation profiles matching',
      'filter': 'from-pub-date:2023-01-01,until-pub-date:2026-09-16', 'rows': 7}),
    ('arxiv_morphology', 'https://export.arxiv.org/api/query',
     {'search_query': '(all:"cell painting" OR all:"morphological profiling") AND (all:"representation" OR all:"drug") AND submittedDate:[202401010000 TO 202609162359]',
      'max_results': 12, 'sortBy': 'submittedDate', 'sortOrder': 'descending'}),
    ('pubmed_morphology', 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi',
     {'db': 'pubmed', 'term': '("Cell Painting"[Title/Abstract] OR "morphological profiling"[Title/Abstract]) AND ("2024/01/01"[Date - Publication] : "2026/09/16"[Date - Publication])',
      'retmode': 'json', 'retmax': 12, 'sort': 'pub date'}),
]


def fetch(job):
    name, url, params = job
    row = dict(source=name, url=url, params=params, utc=datetime.now(timezone.utc).isoformat())
    try:
        response = requests.get(url, params=params, timeout=35)
        path = HERE / (name + '_v1.' + ('xml' if name.startswith('arxiv') else 'json'))
        if path.exists():
            raise ValueError('Preserve source response')
        path.write_bytes(response.content)
        row.update(http_status=response.status_code, file=str(path),
                   sha256=hashlib.sha256(response.content).hexdigest())
        response.raise_for_status()
        if name.startswith('arxiv'):
            ns = {'a': 'http://www.w3.org/2005/Atom'}
            row['items'] = [dict(id=e.findtext('a:id', namespaces=ns),
                title=' '.join(e.findtext('a:title', namespaces=ns).split()),
                published=e.findtext('a:published', namespaces=ns),
                summary=' '.join(e.findtext('a:summary', namespaces=ns).split()))
                for e in ET.fromstring(response.content).findall('a:entry', ns)]
        elif name.startswith('crossref'):
            row['items'] = [{key: item.get(key) for key in
                ('title', 'DOI', 'published', 'container-title', 'abstract', 'link')}
                for item in response.json()['message']['items']]
        else:
            row['pubmed'] = response.json()['esearchresult']
    except Exception as exc:
        row['error'] = str(exc)
    return row


with ThreadPoolExecutor(max_workers=4) as pool:
    records = list(pool.map(fetch, JOBS))
OUT.write_text(json.dumps(records, indent=2) + '\n')
for row in records:
    print(row['source'], row.get('http_status'), row.get('error', ''), flush=True)
    for item in row.get('items', []):
        print(item.get('id', item.get('DOI')), item.get('title'), flush=True)
    if 'pubmed' in row:
        print('PMIDS', row['pubmed'].get('idlist'), flush=True)
