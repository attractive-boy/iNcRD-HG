"""Read-only literature discovery; preserve successful and failed source queries."""
import argparse
import concurrent.futures
import importlib.util
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKILL = Path('/home/zhourongxian/.codex/skills/nature-academic-search/scripts/academic_search.py')
QUERIES = {
    'openalex': [
        ('context', 'tabular foundation model context optimization'),
        ('robust', 'tabular foundation model noisy labels'),
        ('pu', 'tabular foundation model positive unlabeled'),
        ('features', 'tabular foundation model feature adaptation'),
        ('multiobjective', 'multi task learning gradient projection constrained optimization'),
    ],
    'arxiv': [
        ('context', 'all:tabular AND (all:"context optimization" OR all:"context distillation" OR all:"soft prompt" OR all:"learnable context")'),
        ('robust', '(all:TabPFN OR all:TabICL OR all:"tabular foundation") AND (all:noise OR all:robust OR all:unlabeled)'),
        ('adaptation', '(all:TabPFN OR all:TabICL) AND (all:adaptation OR all:retrieval OR all:prompt)'),
        ('constraints', '(all:"multi-task" OR all:"continual learning") AND (all:"gradient projection" OR all:"non-degradation" OR all:"Pareto")'),
    ],
    'pubmed': [
        ('mirna', '(miRNA[Title/Abstract] OR microRNA[Title/Abstract]) AND (drug[Title/Abstract]) AND (association prediction[Title/Abstract] OR resistance prediction[Title/Abstract]) AND (2024:2026[dp])'),
        ('pu', '(positive-unlabeled[Title/Abstract] OR positive unlabeled[Title/Abstract]) AND (drug[Title/Abstract] OR RNA[Title/Abstract]) AND (2024:2026[dp])'),
    ],
}


def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'AcademicLiteratureAudit/1.0'})
    with urllib.request.urlopen(req, timeout=25) as response:
        return response.read().decode('utf-8', errors='replace')


def arxiv(query):
    query += ' AND submittedDate:[202401010000 TO 202609132359]'
    url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({
        'search_query': query, 'start': 0, 'max_results': 35,
        'sortBy': 'relevance', 'sortOrder': 'descending',
    })
    raw = fetch(url)
    root = ET.fromstring(raw)
    ns = {'a': 'http://www.w3.org/2005/Atom', 'x': 'http://arxiv.org/schemas/atom'}
    records = []
    for e in root.findall('a:entry', ns):
        records.append({
            'title': ' '.join(e.findtext('a:title', '', ns).split()),
            'id': e.findtext('a:id', '', ns),
            'authors': [a.findtext('a:name', '', ns) for a in e.findall('a:author', ns)],
            'published': e.findtext('a:published', '', ns),
            'updated': e.findtext('a:updated', '', ns),
            'abstract': ' '.join(e.findtext('a:summary', '', ns).split()),
            'doi': e.findtext('x:doi', '', ns),
            'comment': e.findtext('x:comment', '', ns),
            'journal_ref': e.findtext('x:journal_ref', '', ns),
        })
    return {'url': url, 'raw': raw, 'records': records}


def pubmed(query):
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?' + urllib.parse.urlencode({
        'db': 'pubmed', 'term': query, 'retmode': 'json', 'retmax': 40, 'sort': 'relevance',
    })
    search = json.loads(fetch(url))
    ids = search.get('esearchresult', {}).get('idlist', [])
    if not ids:
        return {'url': url, 'search': search, 'records': []}
    time.sleep(0.4)
    detail_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?' + urllib.parse.urlencode({
        'db': 'pubmed', 'id': ','.join(ids), 'retmode': 'xml',
    })
    raw = fetch(detail_url)
    records = []
    for item in ET.fromstring(raw).findall('.//PubmedArticle'):
        a = item.find('./MedlineCitation/Article')
        title = a.find('ArticleTitle')
        records.append({
            'pmid': item.findtext('./MedlineCitation/PMID'),
            'title': ''.join(title.itertext()) if title is not None else '',
            'authors': [' '.join(filter(None, [x.findtext('ForeName'), x.findtext('LastName')])) for x in a.findall('./AuthorList/Author')],
            'journal': a.findtext('./Journal/Title'),
            'dates': [ET.tostring(x, encoding='unicode') for x in item.findall('.//PubMedPubDate')],
            'abstract': '\n'.join(''.join(x.itertext()) for x in a.findall('./Abstract/AbstractText')),
            'identifiers': {x.get('IdType'): x.text for x in item.findall('./PubmedData/ArticleIdList/ArticleId')},
        })
    return {'url': url, 'detail_url': detail_url, 'search': search, 'raw': raw, 'records': records}


def run_one(source, label, query):
    path = ROOT / f'{source}_{label}.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get('ok'):
            print('CACHED', source, label, len(saved['records']), flush=True)
            return
    record = {'source': source, 'query': query, 'retrieved_utc': datetime.now(timezone.utc).isoformat()}
    try:
        if source == 'openalex':
            spec = importlib.util.spec_from_file_location('academic_search_fallback', SKILL)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            records = module.search(query=query, limit=18, year_from=2024)
            record.update(records=records, via=str(SKILL))
        elif source == 'arxiv':
            record.update(arxiv(query))
        elif source == 'pubmed':
            record.update(pubmed(query))
        record['ok'] = True
    except Exception as exc:
        record.update(ok=False, error=f'{type(exc).__name__}: {exc}', records=[])
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(source, label, 'OK' if record['ok'] else record['error'], flush=True)
    for row in record['records']:
        print(' ', row.get('year', row.get('published', '')), row.get('title'), row.get('id', row.get('doi', row.get('pmid', ''))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', choices=list(QUERIES))
    args = parser.parse_args()
    # Keep each source sequential; arXiv requests are spaced by at least 3 seconds.
    for label, query in QUERIES[args.source]:
        run_one(args.source, label, query)
        if args.source == 'arxiv':
            time.sleep(3.1)
