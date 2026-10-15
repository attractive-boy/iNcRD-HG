"""Read public primary text only; no author datasets or model weights."""
import concurrent.futures
import datetime
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen
HERE = Path(__file__).resolve().parent
JOBS = [('eo_meta', 'https://export.arxiv.org/api/query?id_list=2609.34729v1'),
        ('eo_html', 'https://arxiv.org/html/2609.34729v1'),
        ('popularity_meta', 'https://export.arxiv.org/api/query?id_list=2609.37284v1'),
        ('popularity_html', 'https://arxiv.org/html/2609.37284v1')]


class Text(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts = []; self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in ['script', 'style']:
            self.skip += 1
        if tag in ['p', 'div', 'section', 'h1', 'h2', 'h3', 'h4', 'li', 'tr']:
            self.parts.append('\n')
    def handle_endtag(self, tag):
        if tag in ['script', 'style']:
            self.skip -= 1
    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def fetch(job):
    name, url = job
    r = dict(name=name, url=url, retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    try:
        with urlopen(Request(url, headers={'User-Agent': 'MiDrug-methods-review/1.0'}), timeout=45) as f:
            raw = f.read(); r['status_code'] = f.status
        p = HERE/'raw'/(name + ('.html' if name.endswith('html') else '.xml'))
        p.write_bytes(raw); r.update(status='ok', path=str(p.relative_to(HERE)), sha256=hashlib.sha256(raw).hexdigest())
        if name.endswith('html'):
            parser = Text(); parser.feed(raw.decode())
            p.with_suffix('.txt').write_text('\n'.join(' '.join(s.split()) for s in ''.join(parser.parts).splitlines() if s.strip()))
    except Exception as e:
        r.update(status='error', error=str(e))
    return r


if __name__ == '__main__':
    (HERE/'raw').mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        records = list(pool.map(fetch, JOBS))
    (HERE/'primary_fetch.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))
