"""Public-API literature discovery for the expression-consensus experiment.

No credentials or private data are sent. Search results are evidence for a
bounded prior-art assessment, not proof that no related method exists.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


CROSSREF_QUERIES = (
    'GraphMAE Self-Supervised Masked Graph Autoencoders',
    'HGMAE Heterogeneous Graph Masked Autoencoders',
    'WGCNA an R package for weighted correlation network analysis',
    'ncRNA drug resistance graph contrastive learning',
    'gene coexpression network bootstrap stability consensus',
    'uncertainty weighted consistency regularization graph learning',
    'iNcRD-HG',
)
ARXIV_QUERIES = (
    'all:"graph" AND all:"uncertainty" AND all:"consistency"',
    'all:"gene" AND all:"coexpression" AND all:"bootstrap"',
)
PUBMED_QUERY = '(ncRNA OR miRNA OR lncRNA) AND "drug resistance" AND ("contrastive learning" OR "masked autoencoder" OR "self-supervised")'


def fetch(url):
    request = urllib.request.Request(url, headers={
        'User-Agent': 'expression-consensus-research/1.0',
        'Accept': 'application/json, application/atom+xml, application/xml',
    })
    with urllib.request.urlopen(request, timeout=25) as response:
        return response.read()


def crossref(query):
    params = urllib.parse.urlencode({'query.bibliographic': query, 'rows': 5})
    url = 'https://api.crossref.org/works?' + params
    items = json.loads(fetch(url))['message']['items']
    records = []
    for item in items:
        date = item.get('published', item.get('issued', {})).get('date-parts', [[None]])[0]
        records.append(dict(
            source='CrossRef', query=query, title=' '.join(item.get('title', [])),
            doi=item.get('DOI'), year=date[0],
            authors=[' '.join(filter(None, (a.get('given'), a.get('family')))) for a in item.get('author', [])],
            venue=' '.join(item.get('container-title', [])), url=item.get('URL'),
            abstract=re.sub('<[^>]+>', '', item.get('abstract', '')),
        ))
    return {'source': 'CrossRef', 'query': query, 'url': url, 'records': records}


def arxiv(query):
    params = urllib.parse.urlencode({'search_query': query, 'max_results': 5,
                                    'sortBy': 'relevance'})
    url = 'https://export.arxiv.org/api/query?' + params
    root = ET.fromstring(fetch(url))
    ns = {'a': 'http://www.w3.org/2005/Atom'}
    records = []
    for e in root.findall('a:entry', ns):
        get = lambda key: ' '.join((e.findtext('a:' + key, '', ns)).split())
        records.append(dict(source='arXiv', query=query, title=get('title'),
                            doi=None, year=int(get('published')[:4]) if get('published') else None,
                            authors=[a.findtext('a:name', '', ns) for a in e.findall('a:author', ns)],
                            venue='arXiv preprint', url=get('id'), abstract=get('summary')))
    return {'source': 'arXiv', 'query': query, 'url': url, 'records': records}


def pubmed(query):
    params = urllib.parse.urlencode({'db': 'pubmed', 'term': query, 'retmax': 12, 'retmode': 'json'})
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?' + params
    ids = json.loads(fetch(url))['esearchresult']['idlist']
    records = []
    if ids:
        fetch_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?' + urllib.parse.urlencode(
            {'db': 'pubmed', 'id': ','.join(ids), 'retmode': 'xml'})
        root = ET.fromstring(fetch(fetch_url))
        for article in root.findall('PubmedArticle'):
            title = article.find('.//ArticleTitle')
            identifiers = {e.attrib.get('IdType'): e.text for e in article.findall('./PubmedData/ArticleIdList/ArticleId')}
            year = article.findtext('.//PubDate/Year') or article.findtext('.//ArticleDate/Year')
            records.append(dict(source='PubMed', query=query,
                title=''.join(title.itertext()) if title is not None else '',
                doi=identifiers.get('doi'), pmid=identifiers.get('pubmed'), year=int(year) if year else None,
                authors=[' '.join(filter(None, (a.findtext('ForeName'), a.findtext('LastName')))) for a in article.findall('.//Author')],
                venue=article.findtext('.//Journal/Title'),
                url='https://pubmed.ncbi.nlm.nih.gov/' + identifiers.get('pubmed', '') + '/',
                abstract=' '.join(''.join(a.itertext()) for a in article.findall('.//AbstractText'))))
    return {'source': 'PubMed', 'query': query, 'url': url, 'records': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(crossref, q) for q in CROSSREF_QUERIES] + [(arxiv, q) for q in ARXIV_QUERIES] + [(pubmed, PUBMED_QUERY)]
    responses = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(fn, q): (fn.__name__, q) for fn, q in jobs}
        for future in as_completed(pending):
            source, query = pending[future]
            try:
                response = future.result()
            except Exception as error:
                response = {'source': source, 'query': query, 'error': str(error), 'records': []}
            responses.append(response)
            print(source, len(response['records']), response.get('error', ''), query, flush=True)
    merged = {}
    for response in responses:
        for record in response['records']:
            key = (record.get('doi') or re.sub(r'\W+', '', record['title'])).lower().removeprefix('https://doi.org/')
            if key not in merged or len(record.get('abstract', '')) > len(merged[key].get('abstract', '')):
                merged[key] = record
    payload = {'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
               'scope': 'targeted discovery and abstract/metadata review; not exhaustive novelty certification',
               'mcp_available': False, 'fallback': 'direct public CrossRef, PubMed and arXiv APIs',
               'responses': responses, 'deduplicated': list(merged.values())}
    (out / 'search_results.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    for r in merged.values():
        print(r['source'], r.get('year'), r['title'], r.get('doi') or r['url'], flush=True)


if __name__ == '__main__':
    main()
