"""Small public-API literature discovery batch; no training or data labels."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent


def retrieve(entry):
    name, url, kind = entry
    path = HERE / name
    if path.exists():
        raise ValueError('Preserve earlier retrieval: ' + str(path))
    result = dict(file=str(path), url=url, accessed_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'Scientific-literature-audit/1.0'})
        with urllib.request.urlopen(request, timeout=40) as response:
            content = response.read()
            result.update(status=response.status, content_type=response.headers.get('Content-Type'), bytes=len(content))
        path.write_bytes(content)
        result['sha256'] = hashlib.sha256(content).hexdigest()
        if kind == 'arxiv':
            root = ET.fromstring(content)
            ns = {'a': 'http://www.w3.org/2005/Atom'}
            records = []
            for node in root.findall('a:entry', ns):
                get = lambda key: ' '.join((node.findtext('a:'+key, default='', namespaces=ns)).split())
                records.append(dict(id=get('id'), title=get('title'), published=get('published'), updated=get('updated'),
                    authors=[x.findtext('a:name', namespaces=ns) for x in node.findall('a:author', ns)],
                    abstract=get('summary')))
        else:
            response = json.loads(content)
            records = [dict(title=r.get('title', []), doi=r.get('DOI'), published=r.get('published'),
                authors=r.get('author', []), abstract=r.get('abstract', ''), venue=r.get('container-title', []))
                for r in response['message']['items']]
        result['records'] = records
    except Exception as error:
        result['error'] = repr(error)
    return result


def main():
    destination = HERE / 'PATH_SIGNATURE_DISCOVERY_V1.json'
    if destination.exists():
        raise ValueError('Preserve discovery manifest')
    query = ('((ti:signature OR ti:signatures OR ti:rough) '
             'AND (ti:graph OR abs:"graph neural")) AND submittedDate:[202401010000 TO 202609162359]')
    entries = [('arxiv_path_signature_v1.xml', 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode(
        dict(search_query=query, start=0, max_results=25, sortBy='submittedDate', sortOrder='descending')), 'arxiv')]
    for name, query in [('crossref_graph_signature_v1.json', 'graph path signature neural network'),
                        ('crossref_graph_rough_v1.json', 'graph neural rough path differential signature')]:
        entries.append((name, 'https://api.crossref.org/works?' + urllib.parse.urlencode(
            {'query': query, 'rows': 10, 'filter': 'from-pub-date:2024-01-01,until-pub-date:2026-09-16'}), 'crossref'))
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(retrieve, entries))
    destination.write_text(json.dumps(records, indent=2, ensure_ascii=False)+'\n')
    for source in records:
        print(json.dumps({k: v for k, v in source.items() if k != 'records'}, ensure_ascii=False), flush=True)
        for paper in source.get('records', []):
            print(json.dumps({k: (v[:1200] if k == 'abstract' else v) for k, v in paper.items()
                              if k != 'authors'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
