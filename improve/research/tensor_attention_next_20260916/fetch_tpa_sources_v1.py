"""Public exact-paper metadata and full-text retrieval; no project data read."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib, json
from pathlib import Path
import urllib.request, urllib.parse

HERE = Path(__file__).resolve().parent

def retrieve(item):
    name, url = item
    p = HERE / name
    if p.exists():
        raise ValueError('Preserve existing source')
    r = dict(file=str(p), url=url, accessed_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'Scientific-literature-audit/1.0'})
        with urllib.request.urlopen(request, timeout=40) as response:
            content = response.read()
            r.update(status=response.status, bytes=len(content), content_type=response.headers.get('Content-Type'))
        p.write_bytes(content)
        r['sha256'] = hashlib.sha256(content).hexdigest()
    except Exception as error:
        r['error'] = repr(error)
    return r

def main():
    p = HERE / 'TPA_SOURCE_RETRIEVAL_V1.json'
    if p.exists():
        raise ValueError('Preserve earlier retrieval')
    title = 'Tensor Product Attention Is All You Need'
    entries = [
        ('tpa_metadata_v1.xml', 'https://export.arxiv.org/api/query?id_list=2501.06425'),
        ('tpa_crossref_v1.json', 'https://api.crossref.org/works?' + urllib.parse.urlencode({'query.title': title, 'rows': 5})),
        ('tpa_github_v1.json', 'https://api.github.com/search/repositories?' + urllib.parse.urlencode({'q': '"Tensor Product Attention"', 'per_page': 10})),
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(retrieve, entries))
    p.write_text(json.dumps(rows, indent=2) + '\n')
    for r in rows:
        print(json.dumps(r), flush=True)

if __name__ == '__main__':
    main()
