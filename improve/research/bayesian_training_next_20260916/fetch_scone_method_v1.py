"""Fetch a bounded set of pinned public method files and publisher metadata."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib,json,base64
import requests

here=Path(__file__).resolve().parent
manifest=here/'SCONE_METHOD_SOURCE_V1.json'
if manifest.exists():
 raise ValueError('Preserve source evidence')
tree=json.loads((here/'scone_author_tree_v1.json').read_text())
commit=tree['sha']
by_path={row['path']:row for row in tree['tree']}
paths=['sampling_sgm.py','sde_lib.py','losses_sgm.py','utils_sgm.py','main.py','model/graph/layers.py','util/loss_torch.py']

def fetch(path):
 node=by_path[path]
 url=f'https://raw.githubusercontent.com/jeongwhanchoi/SCONE/{commit}/{path}'
 record=dict(source=path,commit=commit,url=url,utc=datetime.now(timezone.utc).isoformat(),expected_git_blob_sha1=node['sha'])
 target=here/('scone_method_'+path.replace('/','_'))
 try:
  response=requests.get(url,timeout=20)
  record['status']=response.status_code
  response.raise_for_status()
  data=response.content
  sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
  if sha!=node['sha']:
   raise ValueError('Git blob digest mismatch')
  if target.exists():
   if target.read_bytes()!=data:
    raise ValueError('Preserve existing different file')
  else:
   target.write_bytes(data)
  record.update(file=str(target),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),git_blob_verified=True)
 except Exception as error:
  record['error']=str(error)
 return record
with ThreadPoolExecutor(max_workers=4) as pool:
 records=list(pool.map(fetch,paths))
url='https://api.crossref.org/works'
try:
 response=requests.get(url,params={'query.title':'SCONE A Novel Stochastic Sampling to Generate Contrastive Views and Hard Negative Samples for Recommendation','rows':3},timeout=20)
 p=here/'scone_crossref_verification_v1.json'
 p.write_text(response.text)
 records.append(dict(url=response.url,status=response.status_code,file=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),kind='publisher_metadata_search'))
except Exception as error:
 records.append(dict(url=url,error=str(error),kind='publisher_metadata_search'))
manifest.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(records,ensure_ascii=False,indent=2))
