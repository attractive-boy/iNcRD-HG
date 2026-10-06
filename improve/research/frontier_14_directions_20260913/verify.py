"""Verify selected records against Crossref and Europe PMC; retain failures."""
import concurrent.futures
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
DOIS = {
    "incrd": "10.1093/bioinformatics/btag029",
    "mifndra": "10.1093/bib/bbag322",
    "mphgnn": "10.1093/bib/bbag013",
    "plmfusion": "10.1371/journal.pcbi.1013968",
    "spectral_sparse": "10.1109/tcbbio.2026.3729865",
    "dispa": "10.1093/bioinformatics/btag272",
    "scradar": "10.1371/journal.pcbi.1014392",
    "rnaernie": "10.1038/s42256-024-00836-4",
    "rinalmo": "10.1038/s41467-025-60872-5",
    "nucleicbert": "10.1038/s42256-026-01295-9",
    "hydrarna": "10.1186/s13059-025-03853-7",
    "rhofold": "10.1038/s41592-024-02487-0",
    "mirbind2": "10.1093/bioinformatics/btag410",
    "mirformer": "10.1093/bioinformatics/btag222",
    "omicstransformer": "10.1093/bioinformatics/btag468",
    "mirbench": "10.1093/bioinformatics/btaf233",
    "pulsnar": "10.7717/peerj-cs.2451",
    "cpu": "10.1109/tnnls.2026.3706351",
    "perturbnet": "10.1038/s44320-025-00131-3",
    "benchmark2025": "10.1038/s41592-025-02772-6",
    "benchmark2026": "10.1126/sciadv.aed3414",
    "tdmd": "10.1038/s44318-026-00889-8",
    "dcmultitask": "10.1109/jbhi.2026.3717803",
}

def request(url):
    req=urllib.request.Request(url, headers={"User-Agent":"LiteratureReview/1.0", "Accept":"application/json"})
    with urllib.request.urlopen(req, timeout=35) as r:
        return json.load(r)

def get(key, doi, source):
    if source == "crossref":
        url="https://api.crossref.org/works/"+urllib.parse.quote(doi, safe="")
    else:
        url="https://www.ebi.ac.uk/europepmc/webservices/rest/search?"+urllib.parse.urlencode({"query":f'DOI:"{doi}"',"format":"json","resultType":"core","pageSize":5})
    try:
        data=request(url)
        (ROOT/f"{key}_{source}.json").write_text(json.dumps({"url":url,"as_of":"2026-09-13","response":data},ensure_ascii=False,indent=2))
        if source == "crossref":
            x=data['message']
            rec={"title":(x.get('title') or [''])[0],"doi":x.get('DOI'),"venue":(x.get('container-title') or [''])[0],"authors":[a.get('given','')+' '+a.get('family','') for a in x.get('author',[])],"published":x.get('published'),"published_online":x.get('published-online'),"published_print":x.get('published-print'),"type":x.get('type'),"abstract":x.get('abstract'),"relation":x.get('relation'),"update_to":x.get('update-to')}
        else:
            rec=[{k:x.get(k) for k in ['title','doi','id','pmcid','pubYear','firstPublicationDate','abstractText','pubTypeList','fullTextUrlList']} for x in data.get('resultList',{}).get('result',[]) if str(x.get('doi','')).lower()==doi.lower()]
        return {"key":key,"doi":doi,"source":source,"record":rec}
    except Exception as e:
        return {"key":key,"doi":doi,"source":source,"error":str(e)}

def main():
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(get,k,d,s) for k,d in DOIS.items() for s in ['crossref','epmc']]
        for f in concurrent.futures.as_completed(futures):
            r=f.result(); rows.append(r)
            if 'error' in r: print(json.dumps(r),flush=True)
            elif r['source']=='crossref':
                x=r['record']; print(json.dumps({"key":r['key'],"title":x['title'],"venue":x['venue'],"online":x['published_online'],"published":x['published']},ensure_ascii=False),flush=True)
    (ROOT/'verified_sources.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
