"""Read only loss and encoder definitions from the already pinned public commit."""
from fetch_primary import fetch,HERE
import concurrent.futures
import base64
import json

if __name__=='__main__':
    commit=json.loads((HERE/'raw/nodejepa_commit.json').read_text())['sha']
    paths=['graph_jepa/losses/variance.py','graph_jepa/models/encoders.py']
    jobs=[('nodejepa_'+p.rsplit('/',1)[-1]+'.json','https://api.github.com/repos/OliverZ-dot/Node-Jepa/contents/'+p+'?ref='+commit) for p in paths]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(fetch,jobs))
    for row in rows:
        if row['status']=='ok':
            p=HERE/row['file'];j=json.loads(p.read_text());p.with_suffix('').write_bytes(base64.b64decode(j['content']))
    (HERE/'nodejepa_loss_fetch.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
