"""Fetch fixed GitHub commits, then selected source; do not import author code."""
from fetch_primary import fetch, HERE
import base64
import concurrent.futures
import json

if __name__ == '__main__':
    jobs=[(name+'_commit.json','https://api.github.com/repos/'+repo+'/commits/main') for name,repo in [('nodejepa','OliverZ-dot/Node-Jepa'),('vicreg','facebookresearch/vicreg')]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: rows=list(pool.map(fetch,jobs))
    for name,repo in [('nodejepa','OliverZ-dot/Node-Jepa'),('vicreg','facebookresearch/vicreg')]:
        path=HERE/'raw'/(name+'_commit.json')
        if not path.exists():continue
        commit=json.loads(path.read_text())['sha']
        if name=='nodejepa':
            rows.append(fetch((name+'_tree.json','https://api.github.com/repos/'+repo+'/git/trees/'+commit+'?recursive=1')))
        else:
            rows.append(fetch(('vicreg_main_api.json','https://api.github.com/repos/'+repo+'/contents/main_vicreg.py?ref='+commit)))
            p=HERE/'raw/vicreg_main_api.json'
            if p.exists():
                j=json.loads(p.read_text()); (HERE/'raw/vicreg_main.py').write_bytes(base64.b64decode(j['content']))
    (HERE/'pinned_code_fetch.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
