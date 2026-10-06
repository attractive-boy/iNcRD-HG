"""Retrieve public primary sources for a bounded methods survey, not training."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


class Metadata(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.values = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        name = attrs.get('name', '')
        if tag == 'meta' and name.startswith('citation_'):
            self.values.setdefault(name, []).append(attrs.get('content', ''))


def plain(body):
    body = re.sub(r'<(script|style)\b[^>]*>.*?</\1>', '', body, flags=re.S)
    body = re.sub(r'</(?:p|div|section|h[1-6]|tr|li|blockquote)>', '\n', body)
    body = unescape(re.sub(r'<[^>]+>', ' ', body))
    return '\n'.join(' '.join(line.split()) for line in body.splitlines() if line.strip())


def retrieve(aid):
    record = {'arxiv_id': aid, 'url': 'https://arxiv.org/abs/' + aid}
    for kind in ('abs', 'html'):
        url = 'https://arxiv.org/' + kind + '/' + aid
        try:
            req = Request(url, headers={'User-Agent': 'academic-method-survey/1.0'})
            with urlopen(req, timeout=25) as response:
                body = response.read().decode('utf-8')
                record[kind + '_resolved_url'] = response.url
            (ROOT / (aid + '_' + kind + '.html')).write_text(body)
            txt = plain(body)
            (ROOT / (aid + '_' + kind + '.txt')).write_text(txt)
            record[kind + '_chars'] = len(txt)
            if kind == 'abs':
                parser = Metadata()
                parser.feed(body)
                record.update(parser.values)
                abstract = re.search(r'<blockquote[^>]*class="abstract[^>]*>(.*?)</blockquote>', body, re.S)
                record['abstract'] = plain(abstract.group(1)) if abstract else ''
                start = txt.find('Submission history')
                record['history'] = txt[start:start + 1400]
        except Exception as exc:
            record[kind + '_error'] = str(exc)
    return record


if __name__ == '__main__':
    ids = sys.argv[1:] or ['2606.26132', '2606.10071', '2609.11347',
                          '2605.06154', '2510.01510', '2605.11145',
                          '2512.10688', '2608.05016']
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for record in pool.map(retrieve, ids):
            records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != 'history'}, ensure_ascii=False), flush=True)
    payload = {'retrieved_at_utc': datetime.now(timezone.utc).isoformat(), 'records': records}
    target = ROOT / ('primary_fetch.json' if len(sys.argv) == 1 else 'extra_fetch.json')
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
