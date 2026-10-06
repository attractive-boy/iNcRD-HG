"""Retrieve pinned TPA source Git blobs and verify their object IDs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import base64, hashlib, json, urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    dest=HERE/'TPA_AUTHOR_SOURCE_V1.json'
    if dest.exists(): raise ValueError('Preserve author source manifest')
    sha=json.loads((HERE/'tpa_author_commit_v1.json').read_text())['sha']
    tree=json.loads((HERE/'tpa_author_tree_v1.json').read_text())
    assert not tree.get('truncated')
    byname={r['path']:r for r in tree['tree'] if r['type']=='blob'}
    wanted=['README.md','model/T6.py','model/T6_kvonly.py','model/llama-mha.py','config/train_T6_small_adam.py','LICENSE']
    def fetch(name):
        r=byname[name]
        url='https://api.github.com/repos/tensorgi/TPA/git/blobs/'+r['sha']
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Scientific-literature-audit/1.0'}),timeout=40) as response:
            blob=json.load(response)
        content=base64.b64decode(blob['content'])
        actual=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        assert actual==r['sha'] and len(content)==blob['size']
        p=HERE/('tpa_author_v1_'+name.replace('/','__'))
        if p.exists(): raise ValueError('Preserve downloaded source')
        p.write_bytes(content)
        return dict(path=name,file=str(p),url=url,commit=sha,git_blob=r['sha'],git_blob_verified=True,
            sha256=hashlib.sha256(content).hexdigest(),accessed_at_utc=datetime.now(timezone.utc).isoformat())
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(fetch,wanted))
    dest.write_text(json.dumps(rows,indent=2)+'\n')
    for r in rows: print(r['path'],r['sha256'],flush=True)

if __name__=='__main__': main()
