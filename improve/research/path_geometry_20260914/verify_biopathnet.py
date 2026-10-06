from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import requests

OUT=Path(__file__).resolve().parent
DOI='10.1038/s41551-025-01598-z'
JOBS=[('biopathnet_crossref.json','https://api.crossref.org/works/'+DOI,{}),
      ('biopathnet_epmc.json','https://www.ebi.ac.uk/europepmc/webservices/rest/search',dict(query='DOI:'+DOI,format='json',resultType='core')),
      ('biopathnet_nature.html','https://www.nature.com/articles/s41551-025-01598-z',{}),
      ('biopathnet_github.json','https://api.github.com/search/repositories',dict(q='BioPathNet',per_page=5))]


def fetch(job):
    name,url,params=job
    row=dict(file=name,url=url,params=params,requested_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        r=requests.get(url,params=params,headers={'User-Agent':'incrd-literature-audit/1.0'},timeout=(8,25))
        (OUT/name).write_bytes(r.content)
        row.update(status=r.status_code,final_url=r.url,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest())
        r.raise_for_status()
    except Exception as exc:row['error']=str(exc)
    return row


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(fetch,JOBS))
    (OUT/'biopathnet_verification.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))
