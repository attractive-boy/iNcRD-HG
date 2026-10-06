"""Retrieve one OA method and compact JUMP identity metadata at fixed commits."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / 'SELECTED_MORPHOLOGY_FETCH_V1.json'
if OUT.exists():
    raise ValueError('Preserve selected-source fetch')
repo = 'jump-cellpainting/datasets'
commit = '016e865fa0691244e0860943e41c7d6a88ed2580'
tree = {r['path']: r for r in json.loads((HERE / 'datasets_tree_v1.json').read_text())['tree']}
JOBS = []
for path in ['metadata/README.md', 'metadata/compound.csv.gz', 'metadata/crispr.csv.gz',
             'metadata/plate.csv.gz', 'metadata/perturbation_control.csv', 'manifests/profile_index.json']:
    JOBS.append(dict(name='jump_' + path.replace('/', '_'),
        url='https://raw.githubusercontent.com/' + repo + '/' + commit + '/' + path,
        repository=repo, commit=commit, git_blob=tree[path]['sha']))
JOBS += [dict(name='unified_dti_fulltext_v1.xml', url='https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13501321/fullTextXML'),
         dict(name='unified_dti_crossref_v1.json', url='https://api.crossref.org/works/10.1093/bioinformatics/btag368')]


def get(job):
    row = dict(job, utc=datetime.now(timezone.utc).isoformat())
    try:
        path = HERE / job['name']
        if path.exists():
            raise ValueError('Preserve fetched resource')
        response = requests.get(job['url'], timeout=50)
        path.write_bytes(response.content)
        row.update(file=str(path), http_status=response.status_code, bytes=len(response.content),
                   sha256=hashlib.sha256(response.content).hexdigest())
        response.raise_for_status()
        if 'git_blob' in job:
            actual = hashlib.sha1(b'blob ' + str(len(response.content)).encode() + b'\0' + response.content).hexdigest()
            row['git_blob_verified'] = actual == job['git_blob']
            if not row['git_blob_verified']:
                raise ValueError('Git blob mismatch')
    except Exception as exc:
        row['error'] = str(exc)
    return row


def author_repo():
    row = dict(repository='YJRubyLai/Unified-DTI', author_code_executed=False)
    try:
        record = get(dict(name='unified_dti_commits_v1.json', url='https://api.github.com/repos/YJRubyLai/Unified-DTI/commits?per_page=1'))
        row['commit_response'] = record
        if 'error' in record:
            raise ValueError(record['error'])
        sha = json.loads(Path(record['file']).read_text())[0]['sha']
        row['commit'] = sha
        record = get(dict(name='unified_dti_tree_v1.json', url='https://api.github.com/repos/YJRubyLai/Unified-DTI/git/trees/' + sha + '?recursive=1'))
        row['tree_response'] = record
        if 'error' in record:
            raise ValueError(record['error'])
        files = json.loads(Path(record['file']).read_text())['tree']
        readme = next(x for x in files if x['path'].lower() == 'readme.md')
        row['readme'] = get(dict(name='unified_dti_README_v1.md',
            url='https://raw.githubusercontent.com/YJRubyLai/Unified-DTI/' + sha + '/' + readme['path'],
            git_blob=readme['sha'], commit=sha))
    except Exception as exc:
        row['error'] = str(exc)
    return row


with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(get, job) for job in JOBS] + [pool.submit(author_repo)]
    records = [future.result() for future in futures]
OUT.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), records=records,
    association_outcomes_read=False, new_neural_fits=0), indent=2) + '\n')
for row in records:
    print(row.get('name', row.get('repository')), row.get('http_status'), row.get('error', ''), flush=True)
