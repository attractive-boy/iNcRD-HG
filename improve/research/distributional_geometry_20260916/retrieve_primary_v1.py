"""Retrieve identified primary sources and preserve provenance."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent


def fetch(item):
    name, url = item
    path = HERE / name
    if path.exists():
        raise ValueError('Already retrieved: '+name)
    row = dict(url=url, path=str(path), accessed_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'incrd-study/1.0'}), timeout=35) as r:
            raw = r.read()
            row.update(status=r.status, content_type=r.headers.get('Content-Type'))
        path.write_bytes(raw)
        row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        if name.endswith('.pdf'):
            if not raw.startswith(b'%PDF'):
                raise ValueError('Response is not a PDF')
            text = path.with_suffix('.txt')
            subprocess.run(['pdftotext', '-raw', str(path), str(text)], check=True)
            row.update(text=str(text), text_sha256=hashlib.sha256(text.read_bytes()).hexdigest())
    except Exception as e:
        row['error'] = repr(e)
    return row


def main():
    jobs = [
        ('ccp_exact_metadata_v1.json', 'https://api.crossref.org/works/10.1109/TNNLS.2024.3390249'),
        ('ccp_arxiv_search_v1.xml', 'https://export.arxiv.org/api/query?'+urllib.parse.urlencode(
            dict(search_query='ti:"Competitive Covariance Pooling"', max_results=3))),
        ('probabilistic_graph_distance_v2.pdf', 'https://export.arxiv.org/pdf/2401.03913v2'),
        ('covariance_geometry_v3.pdf', 'https://export.arxiv.org/pdf/2407.10484v3'),
        ('gbw_author_commit_v1.json', 'https://api.github.com/repos/jjscc/GBWBN/commits/main'),
    ]
    out = HERE/'PRIMARY_RETRIEVAL_V1.json'
    if out.exists():
        raise ValueError('Preserve previous retrieval')
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, jobs))
    out.write_text(json.dumps(results, indent=2)+'\n')
    for row in results:
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
