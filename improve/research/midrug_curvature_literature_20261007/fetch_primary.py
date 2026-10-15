"""Retrieve public papers only; no training datasets or pretrained models."""
import concurrent.futures,datetime,hashlib,json
from pathlib import Path
import urllib.request
from html.parser import HTMLParser
class TextParser(HTMLParser):
    def __init__(self):super().__init__();self.parts=[];self.skip=0
    def handle_starttag(self,tag,attrs):
        if tag in ["script","style"]:self.skip+=1
    def handle_endtag(self,tag):
        if tag in ["script","style"]:self.skip=max(0,self.skip-1)
    def handle_data(self,data):
        if not self.skip and data.strip():self.parts.append(data.strip())
HERE=Path(__file__).resolve().parent
PAPERS={'ribbon':'2606.27269v1','dak':'2502.10540v1','preconditioners':'2609.25013v1'}
def fetch(item):
    name,aid=item;url=f'https://arxiv.org/html/{aid}'
    r=dict(name=name,arxiv_id=aid,url=url,retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'MiDrug-methods-review/1.0'})
        with urllib.request.urlopen(req,timeout=45) as response:raw=response.read()
        p=HERE/'raw'/f'{name}.html';p.write_bytes(raw)
        parser=TextParser();parser.feed(raw.decode('utf-8'))
        (HERE/f'{name}_primary.txt').write_text('\n'.join(parser.parts))
        r.update(status='ok',sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    except Exception as error:r.update(status='error',error=str(error))
    return r
if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(3) as pool:results=list(pool.map(fetch,PAPERS.items()))
    (HERE/'primary_fetch.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
