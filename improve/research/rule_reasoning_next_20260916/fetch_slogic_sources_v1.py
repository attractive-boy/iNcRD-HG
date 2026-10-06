"""Fetch a small, selected public source set; preserve every response and digest."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import subprocess
import requests

HERE = Path(__file__).resolve().parent
SOURCES = [
    ('slogic_arxiv_metadata_v1.xml', 'https://export.arxiv.org/api/query', {'id_list': '2510.00279'}),
    ('slogic_arxiv_2510.00279v3.pdf', 'https://arxiv.org/pdf/2510.00279v3', None),
    ('slogic_crossref_title_v1.json', 'https://api.crossref.org/works', {'query.title':'SLogic Subgraph-Informed Logical Rule Learning for Knowledge Graph Completion','rows':3}),
]


def fetch(spec):
    name, url, params = spec
    path = HERE / name
    if path.exists():
        raise ValueError('Preserve prior response: ' + str(path))
    row = dict(file=str(path), url=url, params=params,
               utc=datetime.now(timezone.utc).isoformat())
    try:
        response = requests.get(url, params=params, timeout=40,
                                headers={'User-Agent': 'AcademicProjectLiteratureAudit/1.0'})
        row['http_status'] = response.status_code
        row['resolved_url'] = response.url
        path.write_bytes(response.content)
        row['sha256'] = hashlib.sha256(response.content).hexdigest()
        row['bytes'] = len(response.content)
        if name.endswith('.pdf') and response.status_code == 200:
            if not response.content.startswith(b'%PDF'):
                raise ValueError('Not a PDF')
            target = path.with_suffix('.txt')
            if target.exists():
                raise ValueError('Preserve extracted text')
            info = subprocess.check_output(['pdfinfo', str(path)], text=True)
            pages = int(next(line.split(':',1)[1].strip() for line in info.splitlines() if line.startswith('Pages:')))
            subprocess.run(['pdftotext', '-layout', str(path), str(target)], check=True)
            text = target.read_text()
            row.update(pages=pages, text_file=str(target),
                       text_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                       title_present='SLogic' in text,
                       status='open_access_downloaded',
                       si_status='not_requested_not_downloaded')
        elif response.status_code == 200:
            row['status'] = 'metadata_downloaded'
        else:
            row['status'] = 'failed_http'
    except Exception as exc:
        row['error'] = str(exc)
    return row


def main():
    out = HERE / 'SLOGIC_SOURCE_FETCH_V1.json'
    if out.exists():
        raise ValueError('Preserve existing manifest')
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, SOURCES))
    data = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                reason='Selected SLogic from the targeted rule-learning discovery; public OA only.',
                results=results)
    out.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data, indent=2), flush=True)


if __name__ == '__main__':
    main()
