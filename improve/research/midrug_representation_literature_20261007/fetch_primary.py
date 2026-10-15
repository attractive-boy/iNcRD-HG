"""Fetch publicly accessible primary text and metadata for shortlisted methods."""
import concurrent.futures
import datetime
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
JOBS = [
    ('energy_meta.xml', 'https://export.arxiv.org/api/query?id_list=2512.24062'),
    ('energy.html', 'https://arxiv.org/html/2512.24062v2'),
    ('nodejepa_meta.xml', 'https://export.arxiv.org/api/query?id_list=2608.04381'),
    ('nodejepa.html', 'https://arxiv.org/html/2608.04381v1'),
    ('alrtgcl_crossref.json', 'https://api.crossref.org/works/10.1109/TCE.2025.3593065'),
    ('drug_contrastive_pmc.json', 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode(dict(query='DOI:10.1109/TCE.2025.3593065 OR DOI:10.1109/JBHI.2025.3531112 OR DOI:10.1109/JBHI.2026.3717803', format='json', resultType='core'))),
]

class Text(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts = []; self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'): self.skip += 1
        if tag in ('p', 'div', 'section', 'h1', 'h2', 'h3', 'li', 'tr'): self.parts.append('\n')
    def handle_endtag(self, tag):
        if tag in ('script', 'style'): self.skip -= 1
    def handle_data(self, data):
        if not self.skip: self.parts.append(data)

def fetch(job):
    name, url = job
    row = dict(file='raw/' + name, url=url, retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'MiDrug-methods-review/1.0'}), timeout=45) as r:
            data = r.read()
        path = HERE / 'raw' / name; path.parent.mkdir(exist_ok=True); path.write_bytes(data)
        row.update(status='ok', sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        if name.endswith('.html'):
            parser = Text(); parser.feed(data.decode('utf-8'))
            path.with_suffix('.txt').write_text(''.join(parser.parts))
    except Exception as e: row.update(status='error', error=str(e))
    return row

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, JOBS))
    (HERE / 'primary_fetch.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
