"""Small public-source discovery batch; no model code is downloaded/executed."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
QUERIES = {
    'dgm': 'Differentiable Graph Module for Graph Convolutional Networks',
    'dgg': 'DGG Differentiable Graph Generation for Sparse and Adaptive Connectivities',
    'recent_structure': 'graph structure learning differentiable topology',
    'recent_bilevel': 'bilevel graph structure learning link prediction',
}


def search(item):
    name, query = item
    params = {'query.title': query, 'rows': 6,
              'select': 'DOI,title,author,published,container-title,URL,link'}
    if name.startswith('recent_'):
        params['filter'] = 'from-pub-date:2024-01-01,until-pub-date:2026-09-15'
    result = {'source': 'Crossref', 'query': query}
    try:
        response = requests.get('https://api.crossref.org/works', params=params,
                                headers={'User-Agent': 'incrd-hg-research/1.0'}, timeout=35)
        result.update(url=response.url, status=response.status_code)
        if response.ok:
            result['records'] = response.json()['message']['items']
    except requests.RequestException as exc:
        result['error'] = str(exc)
    return name, result


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = dict(pool.map(search, QUERIES.items()))
    path = ROOT / 'LEARNED_STRUCTURE_DISCOVERY.json'
    if path.exists():
        raise RuntimeError('Do not overwrite a discovery record')
    path.write_text(json.dumps({'created_at_utc': datetime.now(timezone.utc).isoformat(),
                               'results': results}, indent=2))
    for name, result in results.items():
        print(name, result.get('status', result.get('error')))
        for r in result.get('records', []):
            print(r.get('DOI'), r.get('title'), r.get('published'))
