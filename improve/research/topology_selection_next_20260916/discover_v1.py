"""Bounded T1 discovery of topological and selected-positive learning methods."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import requests

HERE = Path(__file__).resolve().parent
QUERIES = [
    ('arxiv_topology', 'https://export.arxiv.org/api/query', dict(
        search_query='all:"link prediction" AND (all:"persistent homology" OR all:"topological") AND submittedDate:[202301010000 TO 202609162359]',
        max_results=15, sortBy='submittedDate', sortOrder='descending')),
    ('arxiv_selection', 'https://export.arxiv.org/api/query', dict(
        search_query='(all:"positive unlabeled" OR all:"positive-unlabeled") AND (all:"selected not at random" OR all:"non-random" OR all:"graph") AND submittedDate:[202301010000 TO 202609162359]',
        max_results=15, sortBy='submittedDate', sortOrder='descending')),
    ('crossref_topology', 'https://api.crossref.org/works', dict(
        query='link prediction persistent homology topology', rows=12,
        filter='from-pub-date:2023-01-01,until-pub-date:2026-09-16')),
    ('crossref_selection', 'https://api.crossref.org/works', dict(
        query='positive unlabeled selected not at random graph learning', rows=12,
        filter='from-pub-date:2024-01-01,until-pub-date:2026-09-16')),
]


def get(item):
    name, url, params = item
    record = dict(name=name, url=url, params=params, utc=datetime.now(timezone.utc).isoformat())
    path = HERE / (name + '_v1.' + ('xml' if name.startswith('arxiv') else 'json'))
    if path.exists():
        raise ValueError('Preserve previous discovery')
    try:
        response = requests.get(url, params=params, timeout=30,
            headers={'User-Agent': 'incrd-hg-research/1.0 (bounded academic metadata query)'})
        path.write_bytes(response.content)
        record.update(status=response.status_code, file=str(path),
                      sha256=hashlib.sha256(response.content).hexdigest())
        if response.status_code != 200:
            record['items'] = []
        elif name.startswith('arxiv'):
            ns = {'a': 'http://www.w3.org/2005/Atom'}
            root = ET.fromstring(response.content)
            record['items'] = [dict(title=' '.join(e.findtext('a:title', '', ns).split()),
                id=e.findtext('a:id', '', ns), published=e.findtext('a:published', '', ns),
                authors=[a.findtext('a:name', '', ns) for a in e.findall('a:author', ns)],
                abstract=' '.join(e.findtext('a:summary', '', ns).split())) for e in root.findall('a:entry', ns)]
        else:
            record['items'] = [dict(title=' '.join(e.get('title', [])), doi=e.get('DOI'),
                year=e.get('published', {}).get('date-parts'), journal=e.get('container-title'),
                authors=[(' '.join((a.get('given', ''), a.get('family', '')))).strip() for a in e.get('author', [])],
                abstract=e.get('abstract'), score=e.get('score')) for e in response.json()['message'].get('items', [])]
    except Exception as error:
        record['error'] = repr(error)
    return record


def main():
    dest = HERE / 'TOPOLOGY_SELECTION_DISCOVERY_V1.json'
    if dest.exists():
        raise ValueError('Preserve discovery report')
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(get, QUERIES))
    unique = {}
    for record in records:
        for item in record.get('items', []):
            key = re.sub(r'[^a-z0-9]', '', item['title'].lower())
            if key not in unique:
                unique[key] = dict(item, sources=[record['name']])
            else:
                unique[key]['sources'].append(record['name'])
                if item.get('doi'):
                    unique[key]['doi'] = item['doi']
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), records=records,
        deduplicated=list(unique.values()), all_are_discovery_hits_not_verified_fulltext=True,
        formal_fits=0, selected_papers=[])
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    for row in records:
        print('SOURCE', row['name'], row.get('status'), row.get('error', ''), flush=True)
        for item in row.get('items', []):
            print(item.get('doi', item.get('id')), item.get('year', item.get('published')), item['title'], flush=True)


if __name__ == '__main__':
    main()
