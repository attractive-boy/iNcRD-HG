"""Bounded search for graph equilibrium mechanisms, with raw metadata retained."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import requests


HERE = Path(__file__).resolve().parent
QUERIES = [
    ('arxiv', 'https://export.arxiv.org/api/query', dict(
        search_query='(all:"implicit graph neural" OR all:"equilibrium graph" OR all:"implicit graph learning") AND submittedDate:[202401010000 TO 202609162359]',
        max_results=25, sortBy='submittedDate', sortOrder='descending')),
    ('crossref', 'https://api.crossref.org/works', dict(
        query='implicit graph neural network equilibrium', rows=18,
        filter='from-pub-date:2024-01-01,until-pub-date:2026-09-16')),
    ('openalex', 'https://api.openalex.org/works', dict(
        search='implicit graph neural networks', per_page=18,
        filter='from_publication_date:2024-01-01,to_publication_date:2026-09-16')),
]


def query(spec):
    name, url, params = spec
    record = dict(source=name, url=url, params=params, requested_at_utc=datetime.now(timezone.utc).isoformat())
    path = HERE / (name + '_v1.' + ('xml' if name == 'arxiv' else 'json'))
    if path.exists():
        raise ValueError('Preserve existing metadata')
    try:
        response = requests.get(url, params=params, timeout=30,
            headers={'User-Agent': 'incrd-hg-research/1.0 (academic metadata search)'})
        path.write_bytes(response.content)
        record.update(status=response.status_code, sha256=hashlib.sha256(response.content).hexdigest(), file=str(path))
        response.raise_for_status()
        if name == 'arxiv':
            ns = {'a': 'http://www.w3.org/2005/Atom'}
            record['items'] = [dict(title=' '.join(e.findtext('a:title', '', ns).split()),
                id=e.findtext('a:id', '', ns), published=e.findtext('a:published', '', ns),
                abstract=' '.join(e.findtext('a:summary', '', ns).split()))
                for e in ET.fromstring(response.content).findall('a:entry', ns)]
        elif name == 'crossref':
            record['items'] = [dict(title=' '.join(e.get('title', [])), id=e.get('DOI'),
                published=e.get('published', {}).get('date-parts'), abstract=e.get('abstract'))
                for e in response.json()['message']['items']]
        else:
            record['items'] = [dict(title=e['title'], id=e.get('doi') or e['id'],
                published=e.get('publication_date'), open_access=e.get('open_access'),
                locations=e.get('locations'), abstract_inverted_index=e.get('abstract_inverted_index'))
                for e in response.json()['results']]
    except Exception as error:
        record['error'] = repr(error)
    return record


def main():
    target = HERE / 'EQUILIBRIUM_DISCOVERY_V1.json'
    if target.exists():
        raise ValueError('Preserve existing discovery')
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(query, QUERIES))
    target.write_text(json.dumps(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        records=records, all_hits_require_fulltext_verification=True, formal_fits=0), ensure_ascii=False, indent=2))
    for record in records:
        print(record['source'], record.get('status'), record.get('error', ''), flush=True)
        for item in record.get('items', []):
            print(item['id'], item['published'], item['title'], flush=True)


if __name__ == '__main__':
    main()
