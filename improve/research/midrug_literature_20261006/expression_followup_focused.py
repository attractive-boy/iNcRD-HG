"""Public literature discovery only; no downloaded data enter model inputs."""
import concurrent.futures as cf
import datetime, hashlib, json, re, time
from pathlib import Path
import urllib.parse as up
import urllib.request as ur
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
QUERIES = [
    ('europepmc', '(TITLE_ABS:miRNA OR TITLE_ABS:microRNA) AND TITLE_ABS:drug AND (TITLE_ABS:contrastive OR TITLE_ABS:disentangled OR TITLE_ABS:self-supervised) AND FIRST_PDATE:[2025-01-01 TO 2026-10-06]'),
]


def fetch(job):
    source, query = job
    if source == 'europepmc':
        url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + up.urlencode(dict(query=query,format='json',resultType='core',pageSize=40,sort='FIRST_PDATE_D desc'))
    elif source == 'crossref':
        url = 'https://api.crossref.org/works?' + up.urlencode({'query.title':query,'filter':'from-pub-date:2024-01-01,until-pub-date:2026-10-06','rows':20})
    else:
        url = 'https://export.arxiv.org/api/query?' + up.urlencode(dict(search_query=query,max_results=20,sortBy='submittedDate',sortOrder='descending'))
    result = dict(source=source,query=query,url=url,retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    try:
        req=ur.Request(url,headers={'User-Agent':'MiDrug-methods-review/1.0','Accept':'application/json, application/atom+xml'})
        with ur.urlopen(req,timeout=45) as response: raw=response.read()
        stem=source+'_'+hashlib.sha256(query.encode()).hexdigest()[:10]
        (HERE/'raw').mkdir(exist_ok=True)
        (HERE/'raw'/f'{stem}.txt').write_bytes(raw)
        result['raw_file']=f'raw/{stem}.txt'
        if source=='arxiv':
            ns={'a':'http://www.w3.org/2005/Atom'};tree=ET.fromstring(raw)
            result['papers']=[dict(title=' '.join(e.findtext('a:title',namespaces=ns).split()),id=e.findtext('a:id',namespaces=ns),date=e.findtext('a:published',namespaces=ns),abstract=' '.join(e.findtext('a:summary',namespaces=ns).split()),authors=[a.findtext('a:name',namespaces=ns) for a in e.findall('a:author',ns)]) for e in tree.findall('a:entry',ns)]
        elif source=='europepmc':
            result['papers']=[dict(title=e.get('title'),doi=e.get('doi'),pmid=e.get('id'),pmcid=e.get('pmcid'),date=e.get('firstPublicationDate'),journal=e.get('journalInfo',{}).get('journal',{}).get('title'),abstract=e.get('abstractText'),authors=e.get('authorString')) for e in json.loads(raw)['resultList']['result']]
        else:
            result['papers']=[dict(title=e.get('title',[''])[0],doi=e.get('DOI'),date=e.get('published-online',e.get('published')),journal=e.get('container-title'),abstract=e.get('abstract'),authors=e.get('author')) for e in json.loads(raw)['message']['items']]
        result['status']='ok'
    except Exception as e: result.update(status='error',error=str(e))
    return result

if __name__=='__main__':
    with cf.ThreadPoolExecutor(max_workers=3) as pool: results=list(pool.map(fetch,QUERIES))
    (HERE/'expression_followup_focused.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
    for r in results:
        print(r['source'],r['status'],r.get('error',''),len(r.get('papers',[])))
        for p in r.get('papers',[]): print(p.get('date'),p.get('doi',p.get('id')),p['title'])
