"""Read a small public metadata/source set; no weights or images downloaded."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / 'MORPHOLOGY_SOURCES_V1.json'
if OUT.exists():
    raise ValueError('Preserve source inspection')
discovery = json.loads((HERE / 'MORPHOLOGY_DISCOVERY_V1.json').read_text())
pmids = next(r['pubmed']['idlist'] for r in discovery if 'pubmed' in r)


def save(name, response):
    path = HERE / name
    if path.exists():
        raise ValueError('Preserve fetched resource')
    path.write_bytes(response.content)
    return dict(file=str(path), sha256=hashlib.sha256(response.content).hexdigest(),
                http_status=response.status_code, url=response.url, bytes=len(response.content))


def repository(repo):
    row = dict(repository=repo, read_only=True, code_executed=False)
    label = repo.split('/')[-1]
    try:
        response = requests.get('https://api.github.com/repos/' + repo + '/commits', params={'per_page': 1}, timeout=35)
        row['commit_response'] = save(label + '_commits_v1.json', response)
        response.raise_for_status()
        commit = response.json()[0]['sha']
        row['commit'] = commit
        response = requests.get('https://api.github.com/repos/' + repo + '/git/trees/' + commit, params={'recursive': 1}, timeout=35)
        row['tree_response'] = save(label + '_tree_v1.json', response)
        response.raise_for_status()
        tree = response.json()['tree']
        path = next(x for x in tree if x['path'].lower() == 'readme.md')
        response = requests.get('https://raw.githubusercontent.com/' + repo + '/' + commit + '/' + path['path'], timeout=35)
        row['readme'] = save(label + '_README_v1.md', response)
        response.raise_for_status()
        git_hash = hashlib.sha1(b'blob ' + str(len(response.content)).encode() + b'\0' + response.content).hexdigest()
        if git_hash != path['sha']:
            raise ValueError('Git blob mismatch')
        row['readme']['git_blob_verified'] = True
    except Exception as exc:
        row['error'] = str(exc)
    return row


def metadata(name, url, params):
    row = dict(name=name, requested_url=url, params=params)
    try:
        response = requests.get(url, params=params, timeout=35)
        row.update(save(name, response))
        response.raise_for_status()
    except Exception as exc:
        row['error'] = str(exc)
    return row


with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(repository, repo) for repo in
               ('jump-cellpainting/datasets', 'jump-cellpainting/jump-profiling-recipe', 'microsoft/MorphoHELM')]
    futures += [pool.submit(metadata, 'pubmed_selected_abstracts_v1.xml',
                'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi',
                {'db': 'pubmed', 'id': ','.join(pmids), 'retmode': 'xml'}),
                pool.submit(metadata, 'crossref_jump_title_v1.json', 'https://api.crossref.org/works',
                {'query.title': 'JUMP Cell Painting dataset morphological impact chemical genetic perturbations', 'rows': 6}),
                pool.submit(metadata, 'intervention_arxiv_metadata_v1.xml', 'https://export.arxiv.org/api/query',
                {'id_list': '2604.22832v1'})]
    records = [future.result() for future in futures]
OUT.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), records=records,
                             association_outcomes_read=False, new_neural_fits=0), indent=2) + '\n')
for row in records:
    print(row.get('repository', row.get('name')), row.get('commit'), row.get('http_status'), row.get('error', ''), flush=True)
