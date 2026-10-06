"""Inspect the published external image-DTI training identity table only."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / 'AUTHOR_DTI_TRAIN_FETCH_V1.json'
if OUT.exists():
    raise ValueError('Preserve external training-table fetch')
tree = {x['path']: x for x in json.loads((HERE / 'unified_dti_tree_v1.json').read_text())['tree']}
path = 'src/stage1.2_image/data/train_pairs.csv'
blob = tree[path]
row = dict(utc=datetime.now(timezone.utc).isoformat(), repository='YJRubyLai/Unified-DTI',
    commit=json.loads((HERE / 'unified_dti_commits_v1.json').read_text())[0]['sha'],
    path=path, git_blob=blob['sha'], project_association_outcomes_read=False)
try:
    response = requests.get('https://api.github.com/repos/YJRubyLai/Unified-DTI/git/blobs/' + blob['sha'], timeout=45)
    row['http_status'] = response.status_code
    response.raise_for_status()
    raw = base64.b64decode(response.json()['content'])
    actual = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    if actual != blob['sha'] or len(raw) != blob['size']:
        raise ValueError('External identity-table blob mismatch')
    dest = HERE / 'unified_dti_author_image_train_pairs_v1.csv'
    if dest.exists():
        raise ValueError('Preserve fetched table')
    dest.write_bytes(raw)
    row.update(file=str(dest), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), git_blob_verified=True)
except Exception as exc:
    row['error'] = str(exc)
OUT.write_text(json.dumps(row, indent=2) + '\n')
print(row.get('http_status'), row.get('error', ''), flush=True)
