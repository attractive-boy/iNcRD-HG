"""Fetch actual method paths found in the already pinned author tree."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
OUT = HERE/'SLOGIC_AGGREGATION_SOURCE_V1.json'
if OUT.exists():
    raise ValueError('Preserve method manifest')
parent = json.loads((HERE/'SLOGIC_AUTHOR_SOURCE_V1.json').read_text())
tree = json.loads((HERE/'slogic_author_tree_v1.json').read_text())
commit = parent['commit']
paths = ['run_score_parallel.py']
records = []
for rel in paths:
    item = next(x for x in tree['tree'] if x['path'] == rel)
    target = HERE/('slogic_method_'+rel.replace('/', '_'))
    if target.exists():
        raise ValueError('Preserve method source')
    url = 'https://api.github.com/repos/'+parent['repo']+'/git/blobs/'+item['sha']
    row = dict(source=rel, file=str(target), url=url, commit=commit, git_blob_sha=item['sha'])
    try:
        response = requests.get(url, timeout=30)
        row['http_status'] = response.status_code
        target.with_suffix(target.suffix+'.response.json').write_bytes(response.content)
        response.raise_for_status()
        data = response.json()
        content = base64.b64decode(data['content'])
        expected = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if expected != item['sha'] or data['sha'] != item['sha']:
            raise ValueError('Git blob verification failed')
        target.write_bytes(content)
        row.update(sha256=hashlib.sha256(content).hexdigest(), git_blob_verified=True)
    except Exception as exc:
        row['error'] = str(exc)
    records.append(row)
data = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            parent_sha256=hashlib.sha256((HERE/'SLOGIC_AUTHOR_SOURCE_V1.json').read_bytes()).hexdigest(),
            commit=commit, records=records, author_code_executed=False)
OUT.write_text(json.dumps(data, indent=2)+'\n')
print(json.dumps(data, indent=2), flush=True)
