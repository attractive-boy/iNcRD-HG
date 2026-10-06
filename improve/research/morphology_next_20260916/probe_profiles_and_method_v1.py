"""Inspect profile availability and fixed author method code without executing it."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / 'PROFILE_AND_METHOD_PROBE_V1.json'
if OUT.exists():
    raise ValueError('Preserve probe result')
profiles = json.loads((HERE / 'jump_manifests_profile_index.json').read_text())
author = 'YJRubyLai/Unified-DTI'
commit = json.loads((HERE / 'unified_dti_commits_v1.json').read_text())[0]['sha']
tree = {x['path']: x for x in json.loads((HERE / 'unified_dti_tree_v1.json').read_text())['tree']}


def fetch(name, url, method='GET', blob=None):
    row = dict(name=name, url=url, method=method, utc=datetime.now(timezone.utc).isoformat())
    try:
        response = requests.request(method, url, timeout=40, allow_redirects=True)
        row.update(http_status=response.status_code,
                   headers={key: response.headers.get(key) for key in
                            ('Content-Length', 'Content-Type', 'ETag', 'Accept-Ranges', 'Last-Modified')})
        if method == 'GET':
            path = HERE / name
            if path.exists():
                raise ValueError('Preserve source code')
            path.write_bytes(response.content)
            row.update(file=str(path), sha256=hashlib.sha256(response.content).hexdigest())
            if blob:
                actual = hashlib.sha1(b'blob ' + str(len(response.content)).encode() + b'\0' + response.content).hexdigest()
                row.update(commit=commit, git_blob=blob, git_blob_verified=actual == blob)
                if actual != blob:
                    raise ValueError('Git blob mismatch')
        response.raise_for_status()
    except Exception as exc:
        row['error'] = str(exc)
    return row


with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(fetch, 'profile_' + row['subset'], row['url'], 'HEAD')
               for row in profiles if row['subset'] in ('compound', 'crispr')]
    for path in ['src/stage1.2_image/model/config.py', 'src/stage1.2_image/model/model.py',
                 'src/stage1.2_image/model/dataset.py', 'src/stage1.2_image/model/loss.py',
                 'src/stage1.2_image/model/train.py', 'src/stage2/model/loss.py', 'tutorial/Image.ipynb']:
        futures.append(pool.submit(fetch, 'unified_' + path.replace('/', '_'),
            'https://raw.githubusercontent.com/' + author + '/' + commit + '/' + path, 'GET', tree[path]['sha']))
    recipe = 'https://raw.githubusercontent.com/broadinstitute/jump-profiling-recipe/a917fa79342ff92cf0ea05d6d9174d9028a90f8f/'
    for path in ['README.md', 'inputs/compound.json', 'inputs/crispr.json']:
        futures.append(pool.submit(fetch, 'recipe_' + path.replace('/', '_'), recipe + path))
    records = [future.result() for future in futures]
OUT.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), records=records,
    author_repository=author, author_commit=commit, author_code_executed=False, neural_fits=0), indent=2) + '\n')
for row in records:
    print(row['name'], row.get('http_status'), row.get('error', ''), row.get('headers', {}).get('Content-Length'), flush=True)
