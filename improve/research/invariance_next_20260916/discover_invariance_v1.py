from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import requests,json,hashlib,xml.etree.ElementTree as ET
h=Path(__file__).resolve().parent
p=h/'DISCOVERY_V1.json'
if p.exists():
 raise ValueError('Preserve literature discovery')
jobs=[('arxiv_link','https://export.arxiv.org/api/query',{'search_query':'all:"link prediction" AND (all:invariance OR all:"domain generalization" OR all:"causal learning")','start':0,'max_results':12,'sortBy':'submittedDate','sortOrder':'descending'}),('arxiv_risk','https://export.arxiv.org/api/query',{'search_query':'all:"risk distribution matching" OR ti:Fishr','start':0,'max_results':8}),('crossref_risk','https://api.crossref.org/works',{'query.title':'Domain Generalization with Risk Distribution Matching','rows':5}),('crossref_causal_link','https://api.crossref.org/works',{'query':'causal invariant link prediction graph domain generalization','filter':'from-pub-date:2024-01-01','rows':6})]

def fetch(job):
 name,url,params=job
 record=dict(name=name,url=url,params=params,utc=datetime.now(timezone.utc).isoformat())
 try:
  response=requests.get(url,params=params,timeout=25)
  record['status']=response.status_code
  target=h/(name+'_v1.'+('xml' if name.startswith('arxiv') else 'json'))
  target.write_bytes(response.content)
  record.update(file=str(target),sha256=hashlib.sha256(response.content).hexdigest())
  response.raise_for_status()
  if name.startswith('arxiv'):
   ns={'a':'http://www.w3.org/2005/Atom'}
   entries=ET.fromstring(response.content).findall('a:entry',ns)
   record['results']=[dict(id=e.findtext('a:id',namespaces=ns),title=' '.join(e.findtext('a:title',namespaces=ns).split()),published=e.findtext('a:published',namespaces=ns),summary=' '.join(e.findtext('a:summary',namespaces=ns).split())) for e in entries]
  else:
   record['results']=[{k:x.get(k) for k in ('title','DOI','published','container-title','abstract')} for x in response.json()['message']['items']]
 except Exception as error:
  record['error']=str(error)
 return record
with ThreadPoolExecutor(max_workers=4) as pool:
 results=list(pool.map(fetch,jobs))
p.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(results,ensure_ascii=False,indent=2))
