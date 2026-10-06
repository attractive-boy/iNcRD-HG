"""Read only three relevant files from the paper-linked repository at a pinned commit."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = 'lehgtrung/slogic-kdir-2026'
OUT = HERE/'SLOGIC_AUTHOR_SOURCE_V1.json'
if OUT.exists():
    raise ValueError('Preserve author-source manifest')
records = []


def api(suffix, name):
    path = HERE/name
    if path.exists():
        raise ValueError('Preserve source response')
    url = 'https://api.github.com/repos/' + REPO + '/' + suffix
    response = requests.get(url, timeout=30)
    path.write_bytes(response.content)
    records.append(dict(url=url, file=str(path), http_status=response.status_code,
                        sha256=hashlib.sha256(response.content).hexdigest()))
    response.raise_for_status()
    return response.json()


data = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), repo=REPO,
            discovery='Paper first-page footnote links this author repository',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            code_executed=False, records=records)
try:
    commit = api('commits?per_page=1', 'slogic_author_commits_v1.json')[0]['sha']
    data['commit'] = commit
    tree = api('git/trees/'+commit+'?recursive=1', 'slogic_author_tree_v1.json')
    data['tree_truncated'] = tree.get('truncated', False)
    selected = [x for x in tree['tree'] if x['type'] == 'blob' and
                x['path'] in ('README.md', 'model.py', 'models.py', 'train.py')]
    data['selected_paths'] = [x['path'] for x in selected]
    data['python_paths'] = [x['path'] for x in tree['tree'] if x['path'].endswith('.py')]
    data['files'] = []
    for item in selected:
        name = 'slogic_author_' + item['path'].replace('/', '_')
        blob = api('git/blobs/'+item['sha'], name+'.blob.json')
        content = base64.b64decode(blob['content'])
        actual = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if actual != item['sha'] or blob['sha'] != item['sha']:
            raise ValueError('Git blob mismatch')
        path = HERE/name
        if path.exists():
            raise ValueError('Preserve code')
        path.write_bytes(content)
        data['files'].append(dict(path=item['path'], file=str(path), commit=commit,
                                 sha256=hashlib.sha256(content).hexdigest(),
                                 git_blob_sha=item['sha'], git_blob_verified=True))
except Exception as exc:
    data['error'] = str(exc)
OUT.write_text(json.dumps(data, indent=2)+'\n')
print(json.dumps(data, indent=2), flush=True)
