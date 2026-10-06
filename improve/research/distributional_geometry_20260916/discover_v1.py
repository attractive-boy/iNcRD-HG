"""Public-source discovery for distributional graph representations, without task labels."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent


def fetch(spec):
    name, url, kind = spec
    path = HERE / name
    if path.exists():
        raise ValueError('Preserve earlier retrieval: ' + str(path))
    result = dict(url=url, accessed_at_utc=datetime.now(timezone.utc).isoformat(), file=str(path))
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'incrd-literature-study/1.0'})
        with urllib.request.urlopen(request, timeout=35) as response:
            content = response.read()
            result.update(status=response.status, content_type=response.headers.get('Content-Type'))
        path.write_bytes(content)
        result.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        if kind == 'arxiv':
            ns = {'a': 'http://www.w3.org/2005/Atom'}
            records = []
            for node in ET.fromstring(content).findall('a:entry', ns):
                get = lambda key: ' '.join(node.findtext('a:'+key, default='', namespaces=ns).split())
                records.append(dict(id=get('id'), title=get('title'), published=get('published'),
                    updated=get('updated'), abstract=get('summary'),
                    authors=[n.findtext('a:name', namespaces=ns) for n in node.findall('a:author', ns)]))
        else:
            records = [dict(title=r.get('title', []), doi=r.get('DOI'),
                published=r.get('published'), authors=r.get('author', []),
                abstract=r.get('abstract'), venue=r.get('container-title', []))
                for r in json.loads(content)['message']['items']]
        result['records'] = records
    except Exception as error:
        result['error'] = repr(error)
    return result


def main():
    dest = HERE / 'DISCOVERY_V1.json'
    if dest.exists():
        raise ValueError('Preserve earlier discovery')
    aq = ('(ti:graph AND (ti:Gaussian OR ti:Wasserstein OR ti:covariance OR ti:distributional)) '
          'AND submittedDate:[202401010000 TO 202609162359]')
    jobs = [('arxiv_distributional_v1.xml', 'https://export.arxiv.org/api/query?'+urllib.parse.urlencode(
        dict(search_query=aq, start=0, max_results=40, sortBy='submittedDate', sortOrder='descending')), 'arxiv')]
    for name, query in [('crossref_covariance_v1.json', 'graph neural network covariance pooling'),
                        ('crossref_wasserstein_v1.json', 'Wasserstein Gaussian graph link prediction')]:
        jobs.append((name, 'https://api.crossref.org/works?'+urllib.parse.urlencode(
            {'query': query, 'rows': 15, 'filter': 'from-pub-date:2024-01-01,until-pub-date:2026-09-16'}), 'crossref'))
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(fetch, jobs))
    dest.write_text(json.dumps(records, indent=2, ensure_ascii=False)+'\n')
    for source in records:
        print(json.dumps({k:v for k,v in source.items() if k != 'records'}), flush=True)
        for paper in source.get('records', []):
            print(json.dumps({k:v for k,v in paper.items() if k in ('id','title','doi','published')},
                             ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
