"""Fetch a bounded list of public arXiv full texts, with content verification."""
import argparse
import hashlib
import html
import json
import re
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class ExtractText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.skip += 1
        if tag in ('p', 'div', 'section', 'h1', 'h2', 'h3', 'h4', 'li', 'tr'):
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.skip = max(0, self.skip - 1)
        if tag in ('p', 'div', 'section', 'h1', 'h2', 'h3', 'h4', 'li', 'tr'):
            self.parts.append('\n')

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

    def value(self):
        return '\n'.join(line for row in ''.join(self.parts).splitlines()
                         if (line := ' '.join(row.split())))


def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'AcademicLiteratureAudit/1.0'})
    with urllib.request.urlopen(req, timeout=35) as response:
        data = response.read()
        return data, {'url': url, 'final_url': response.url, 'http_status': response.status,
                      'content_type': response.headers.get('Content-Type'), 'bytes': len(data),
                      'sha256': hashlib.sha256(data).hexdigest()}


def get_paper(paper):
    slug, arxiv_id, expected_title = paper['slug'], paper['arxiv_id'], paper['title']
    manifest_file = ROOT / f'{slug}_fulltext_manifest.json'
    if manifest_file.exists():
        old = json.loads(manifest_file.read_text())
        if old.get('status') in ('full_text_html_available', 'open_access_downloaded'):
            print(slug, 'CACHED', old['status'], flush=True)
            return old
    record = {**paper, 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
              'attempts': [], 'si_status': 'not_requested_not_downloaded'}
    try:
        raw, info = fetch(f'https://arxiv.org/abs/{arxiv_id}')
        record['attempts'].append(info)
        page = raw.decode('utf-8', errors='replace')
        (ROOT / f'{slug}_abs.html').write_text(page)
        metadata = {}
        for key, value in re.findall(r'<meta name="(citation_[^"]+)" content="([^"]*)"', page):
            metadata.setdefault(key, []).append(html.unescape(value))
        record['metadata'] = metadata
        found = metadata.get('citation_title', [''])[0]
        normal = lambda s: re.sub(r'[^a-z0-9]+', '', s.lower())
        if normal(expected_title) != normal(found):
            raise ValueError(f'Title mismatch: expected {expected_title!r}, got {found!r}')
        record['title_verified'] = True
        versions = re.findall(r'\[v(\d+)\].*?(?=\[v|</div>)', page, re.S)
        record['version_numbers_seen'] = versions
    except Exception as exc:
        record.update(status='url_needs_repair', error=str(exc))
        manifest_file.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        print(slug, record['status'], str(exc), flush=True)
        return record

    try:
        raw, info = fetch(f'https://arxiv.org/html/{arxiv_id}')
        record['attempts'].append(info)
        page = raw.decode('utf-8', errors='replace')
        parser = ExtractText()
        parser.feed(page)
        text = parser.value()
        if len(text) < 12000 or 'ltx_document' not in page:
            raise ValueError('Response does not contain a full arXiv article')
        (ROOT / 'fulltext' / f'{slug}.html').write_text(page)
        (ROOT / 'fulltext' / f'{slug}.txt').write_text(text + '\n')
        record.update(status='full_text_html_available', fulltext_path=f'fulltext/{slug}.txt',
                      text_characters=len(text), pdf_status='not_requested_html_sufficient')
    except Exception as exc:
        record['attempts'].append({'url': f'https://arxiv.org/html/{arxiv_id}', 'error': str(exc)})
        try:
            raw, info = fetch(f'https://arxiv.org/pdf/{arxiv_id}')
            record['attempts'].append(info)
            if not raw.startswith(b'%PDF'):
                raise ValueError('Response lacks PDF signature')
            pdf = ROOT / 'PDFs' / f'{slug}_{arxiv_id}.pdf'
            pdf.write_bytes(raw)
            target = ROOT / 'fulltext' / f'{slug}.txt'
            if shutil.which('pdftotext'):
                subprocess.run(['pdftotext', '-layout', str(pdf), str(target)], check=True,
                               capture_output=True, timeout=30)
                text = target.read_text(errors='replace')
                pages = max(1, text.count('\f'))
            else:
                import fitz
                with fitz.open(pdf) as document:
                    pages = len(document)
                    text = '\n'.join(page.get_text() for page in document)
                target.write_text(text)
            if len(text) < 3000:
                raise ValueError('PDF text too short to verify')
            expected_words = set(re.findall(r'[a-z]+', expected_title.lower()))
            front_words = set(re.findall(r'[a-z]+', text[:8000].lower()))
            coverage = len(expected_words & front_words) / max(len(expected_words), 1)
            if coverage < .85:
                raise ValueError(f'PDF title check failed ({coverage:.3f})')
            record.update(status='open_access_downloaded', pdf_path=str(pdf.relative_to(ROOT)),
                          fulltext_path=str(target.relative_to(ROOT)), page_count=pages,
                          text_characters=len(text), title_token_coverage=coverage)
        except Exception as pdf_exc:
            record.update(status='failed_after_retry', error=str(pdf_exc))
    manifest_file.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(slug, record['status'], record.get('text_characters', record.get('error')), flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('list_file')
    args = parser.parse_args()
    (ROOT / 'fulltext').mkdir(exist_ok=True)
    (ROOT / 'PDFs').mkdir(exist_ok=True)
    papers = json.loads(Path(args.list_file).read_text())
    if len(papers) > 10:
        raise ValueError('Use a bounded batch of at most 10 selected papers')
    for paper in papers:
        get_paper(paper)
        time.sleep(1)
