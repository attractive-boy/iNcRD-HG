"""One fixed-blob API retry for the three timed-out author source files."""
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / 'AUTHOR_SOURCES_RETRY_V2.json'
if OUT.exists():
    raise ValueError('Preserve author retry')
tree = {x['path']: x for x in json.loads((HERE / 'unified_dti_tree_v1.json').read_text())['tree']}
commit = json.loads((HERE / 'unified_dti_commits_v1.json').read_text())[0]['sha']


def fetch(path):
    item = tree[path]
    row = dict(path=path, commit=commit, git_blob=item['sha'])
    try:
        url = 'https://api.github.com/repos/YJRubyLai/Unified-DTI/git/blobs/' + item['sha']
        response = requests.get(url, timeout=40)
        row.update(url=url, http_status=response.status_code)
        response.raise_for_status()
        raw = base64.b64decode(response.json()['content'])
        actual = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if actual != item['sha'] or len(raw) != item['size']:
            raise ValueError('Incorrect Git blob')
        dest = HERE / ('unified_v2_' + path.replace('/', '_'))
        if dest.exists():
            raise ValueError('Preserve source file')
        dest.write_bytes(raw)
        row.update(file=str(dest), sha256=hashlib.sha256(raw).hexdigest(), git_blob_verified=True)
    except Exception as exc:
        row['error'] = str(exc)
    print(path, row.get('http_status'), row.get('error', ''), flush=True)
    return row


with ThreadPoolExecutor(max_workers=3) as pool:
    records = list(pool.map(fetch, ['README.md', 'src/stage1.2_image/model/config.py', 'tutorial/Image.ipynb']))
OUT.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), records=records,
    author_code_executed=False), indent=2) + '\n')
