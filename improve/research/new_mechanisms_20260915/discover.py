"""Public-metadata discovery; bounded requests and preserved source responses."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
QUERIES = {
    'block': '(ti:"block model" OR ti:"blockmodel" OR ti:"mixed membership") AND (all:"prediction" OR all:"covariate")',
    'transport': 'all:"optimal transport" AND (ti:"link prediction" OR ti:"graph" OR ti:"network")',
    'geometry': '(ti:"hyperbolic" OR ti:"Lorentz") AND (all:"link prediction" OR all:"drug")',
    'multitask': '(ti:"multi-task" OR ti:"multitask") AND (all:"gradient" OR all:"orthogonal")',
    'relational': '(ti:"relational" OR ti:"link prediction") AND (all:"Bayesian" OR all:"uncertainty")',
}


def fetch(name, query):
    query = '(' + query + ') AND submittedDate:[202401010000 TO 202609152359]'
    url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode(dict(
        search_query=query, start=0, max_results=12, sortBy='submittedDate', sortOrder='descending'))
    row = dict(name=name, url=url, accessed_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'incrd-hg-research/1.0'})
        with urllib.request.urlopen(request, timeout=40) as response:
            raw = response.read()
        path = ROOT / (name + '_arxiv.xml')
        path.write_bytes(raw)
        row.update(status='ok', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        root = ET.fromstring(raw)
        ns = {'a': 'http://www.w3.org/2005/Atom'}
        row['papers'] = [{key: ' '.join((entry.findtext('a:' + key, '', ns)).split())
                          for key in ('id', 'title', 'published', 'summary')}
                         for entry in root.findall('a:entry', ns)]
    except Exception as error:
        row.update(status='error', error=str(error))
    return row


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda item: fetch(*item), QUERIES.items()))
    (ROOT / 'discovery.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    for row in rows:
        print(row['name'], row['status'])
        for paper in row.get('papers', []):
            print(paper['published'][:10], paper['id'], paper['title'])
