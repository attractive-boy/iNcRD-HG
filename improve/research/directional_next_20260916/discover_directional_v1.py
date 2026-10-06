"""Bounded multi-source discovery of recent directional spectral mechanisms."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
QUERIES = [
    ('arxiv_directional_v1', 'https://export.arxiv.org/api/query', dict(
        search_query='(all:"magnetic Laplacian" OR all:"magnetic graph" OR all:"directed graph neural") AND submittedDate:[202401010000 TO 202609162359]',
        start=0, max_results=25, sortBy='submittedDate', sortOrder='descending')),
    ('crossref_directional_v1', 'https://api.crossref.org/works', dict(
        query='magnetic graph neural network directed adaptive', rows=15,
        filter='from-pub-date:2024-01-01,until-pub-date:2026-09-16')),
    ('openalex_directional_v1', 'https://api.openalex.org/works', dict(
        search='magnetic graph neural network', per_page=20,
        filter='from_publication_date:2024-01-01,to_publication_date:2026-09-16')),
]


def fetch(item):
    name, endpoint, params = item
    url = endpoint + '?' + urllib.parse.urlencode(params)
    dest = HERE / (name + ('.xml' if name.startswith('arxiv') else '.json'))
    if dest.exists():
        raise ValueError('Preserve earlier search artifacts')
    record = dict(name=name, url=url)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={
                'User-Agent': 'incrd-hg-literature-audit/1.0', 'Accept': '*/*'}), timeout=30) as response:
            raw = response.read()
            record.update(status=response.status, content_type=response.headers.get('Content-Type', ''))
        dest.write_bytes(raw)
        record.update(path=str(dest.resolve()), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        if name.startswith('arxiv'):
            root = ET.fromstring(raw)
            ns = {'a': 'http://www.w3.org/2005/Atom'}
            values = [dict(title=' '.join(e.findtext('a:title', '', ns).split()),
                id=e.findtext('a:id', '', ns), date=e.findtext('a:published', '', ns),
                abstract=' '.join(e.findtext('a:summary', '', ns).split()),
                authors=[a.findtext('a:name', '', ns) for a in e.findall('a:author', ns)])
                for e in root.findall('a:entry', ns)]
        elif name.startswith('crossref'):
            values = [dict(title=' '.join(e.get('title', [])), doi=e.get('DOI'),
                date=e.get('published', {}), authors=e.get('author', []),
                abstract=e.get('abstract', ''), link=e.get('link', []))
                for e in json.loads(raw)['message']['items']]
        else:
            values = []
            for e in json.loads(raw)['results']:
                inverse = e.get('abstract_inverted_index') or {}
                tokens = {p: w for w, positions in inverse.items() for p in positions}
                values.append(dict(title=e.get('title'), doi=e.get('doi'), date=e.get('publication_date'),
                    id=e.get('id'), abstract=' '.join(tokens[p] for p in sorted(tokens)),
                    authors=[a['author']['display_name'] for a in e.get('authorships', [])],
                    locations=e.get('locations', [])))
        record['records'] = values
        record['valid_response'] = True
    except Exception as error:
        record.update(error=repr(error), valid_response=False)
    print(json.dumps({k:v for k,v in record.items() if k != 'records'}), flush=True)
    return record


if __name__ == '__main__':
    output = HERE / 'DIRECTIONAL_DISCOVERY_V1.json'
    if output.exists():
        raise ValueError('Preserve earlier discovery output')
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, QUERIES))
    output.write_text(json.dumps(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        academic_mcp_available=False, workflow='multi-source-search',
        scope='Bounded discovery; no novelty or implementation claims; no project outcome arrays read',
        records=results), indent=2, ensure_ascii=False))
    for source in results:
        for item in source.get('records', []):
            print(source['name'], item['title'], item.get('doi', item.get('id')), item['date'], flush=True)
