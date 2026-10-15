"""Fetch primary metadata and open HTML, with explicit retrieval provenance."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
from html.parser import HTMLParser


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.text = []; self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ['script', 'style', 'nav']: self.skip += 1
        if tag in ['p', 'h1', 'h2', 'h3', 'h4', 'section']: self.text.append('\n')

    def handle_endtag(self, tag):
        if tag in ['script', 'style', 'nav']: self.skip -= 1

    def handle_data(self, data):
        if not self.skip: self.text.append(data)

HERE = Path(__file__).resolve().parent
JOBS = [
    ('averaging_crossref.json', 'https://api.crossref.org/works/10.1145%2F3731443.3771365'),
    ('averaging_arxiv.xml', 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode(dict(search_query='ti:"Parameter Averaging in Link Prediction"', max_results=5))),
    ('auxiliary_arxiv.xml', 'https://export.arxiv.org/api/query?id_list=2609.29774'),
    ('auxiliary.html', 'https://arxiv.org/html/2609.29774v1'),
]


def fetch(job):
    name, url = job
    row = dict(url=url, file=name, retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'MiDrug-methods-review/1.0'})
        with urllib.request.urlopen(request, timeout=40) as response:
            data = response.read(); row['resolved_url'] = response.url
        (HERE / 'raw' / name).write_bytes(data)
        row.update(status='ok', sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        if name.endswith('.html'):
            parsed = TextParser(); parsed.feed(data.decode('utf-8'))
            (HERE / 'raw' / name.replace('.html', '.txt')).write_text(' '.join(parsed.text))
    except Exception as exc: row.update(status='error', error=str(exc))
    return row


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: rows = list(pool.map(fetch, JOBS))
    (HERE / 'primary_fetch.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
