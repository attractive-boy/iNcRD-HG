"""Fetch a small set of public full texts for method/limitation inspection."""
import concurrent.futures
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent

class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts=[]; self.skip=0
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style'): self.skip+=1
        if tag in ('p','h1','h2','h3','h4','li','section'): self.parts.append('\n')
    def handle_endtag(self,tag):
        if tag in ('script','style') and self.skip: self.skip-=1
        if tag in ('p','h1','h2','h3','h4','li'): self.parts.append('\n')
    def handle_data(self,data):
        if not self.skip: self.parts.append(data)

def get(key,url,fmt):
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'LiteratureReview/1.0'})
        with urllib.request.urlopen(req,timeout=35) as r:
            raw=r.read(8000000); status=r.status
        (ROOT/f'{key}_full.{fmt}').write_bytes(raw)
        if fmt=='xml':
            tree=ET.fromstring(raw)
            lines=[' '.join(e.itertext()) for e in tree.iter() if e.tag in ('article-title','title','p')]
        else:
            parser=TextParser(); parser.feed(raw.decode('utf-8',errors='replace'))
            lines=parser.parts
        (ROOT/f'{key}_full.txt').write_text('\n'.join(' '.join(x.split()) for x in '\n'.join(lines).splitlines() if x.strip()))
        return {'key':key,'url':url,'status':status,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'file':f'{key}_full.{fmt}'}
    except Exception as e:
        return {'key':key,'url':url,'error':str(e)}

def main():
    verified=json.loads((ROOT/'verified_sources.json').read_text())
    requests=[]
    for key in ['plmfusion','mirbind2','mirformer','mirbench','rinalmo','perturbnet']:
        row=next(x for x in verified if x['key']==key and x['source']=='epmc')
        rec=next((x for x in row.get('record',[]) if x.get('pmcid')),None)
        if rec: requests.append((key,'https://www.ebi.ac.uk/europepmc/webservices/rest/'+rec['pmcid']+'/fullTextXML','xml'))
    requests.extend([
        ('nucleicbert','https://www.nature.com/articles/s42256-026-01295-9','html'),
        ('benchmark2026','https://www.science.org/doi/10.1126/sciadv.aed3414','html'),
    ])
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(lambda a:get(*a),requests):
            rows.append(row); print(json.dumps(row),flush=True)
    (ROOT/'reading_manifest.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__': main()
