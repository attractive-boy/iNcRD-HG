"""One API-blob retry for JUMP raw-download timeouts; retain the failed manifest."""
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / 'JUMP_METADATA_RETRY_V2.json'
if OUT.exists():
    raise ValueError('Preserve retry record')
repo = 'jump-cellpainting/datasets'
commit = '016e865fa0691244e0860943e41c7d6a88ed2580'
tree = {x['path']: x for x in json.loads((HERE / 'datasets_tree_v1.json').read_text())['tree']}


def fetch(path):
    blob = tree[path]
    row = dict(path=path, repository=repo, commit=commit, git_blob=blob['sha'], expected_bytes=blob['size'])
    try:
        url = 'https://api.github.com/repos/' + repo + '/git/blobs/' + blob['sha']
        response = requests.get(url, timeout=50)
        row.update(url=url, http_status=response.status_code)
        response.raise_for_status()
        value = response.json()
        if value['encoding'] != 'base64':
            raise ValueError('Unexpected blob encoding')
        raw = base64.b64decode(value['content'])
        actual = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if actual != blob['sha'] or len(raw) != blob['size']:
            raise ValueError('Blob verification failed')
        target = HERE / ('jump_v2_' + path.replace('/', '_'))
        if target.exists():
            raise ValueError('Preserve retry resource')
        target.write_bytes(raw)
        row.update(file=str(target), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), git_blob_verified=True)
    except Exception as exc:
        row['error'] = str(exc)
    print(path, row.get('http_status'), row.get('error', ''), flush=True)
    return row


with ThreadPoolExecutor(max_workers=3) as pool:
    records = list(pool.map(fetch, ['metadata/compound.csv.gz', 'metadata/crispr.csv.gz',
                                   'metadata/plate.csv.gz', 'metadata/perturbation_control.csv']))
OUT.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), records=records,
    previous_manifest_sha256=hashlib.sha256((HERE / 'SELECTED_MORPHOLOGY_FETCH_V1.json').read_bytes()).hexdigest()), indent=2) + '\n')
