"""Task-focused public literature search; never imported by model training."""
import concurrent.futures,datetime,hashlib,importlib.util,json,re
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'midrug_literature_20261006/expression_followup_search.py'
spec=importlib.util.spec_from_file_location('task_public_fetch',SOURCE)
fetcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(fetcher);fetcher.HERE=HERE
QUERIES=[
    ('arxiv','all:"link prediction" AND (all:"higher-order" OR all:"subgraph" OR all:"pairwise") AND submittedDate:[202501010000 TO 202610062359]'),
    ('arxiv','all:"tabular" AND (all:"retrieval" OR all:"nearest neighbor") AND submittedDate:[202501010000 TO 202610062359]'),
    ('crossref','link prediction generalization training graph augmentation'),
]

def norm_title(x):
    return ' '.join(w for w in re.findall(r'\w+',x.lower()) if w not in {'a','an','the','in','of','for','on','to','and','with','by','et','al'})
def surname(p):
    a=p.get('authors') or []
    if isinstance(a,str):return a.split(',')[0].split()[0].lower() if a else ''
    if isinstance(a,list) and a:
        first=a[0]
        return first.get('family','').lower() if isinstance(first,dict) else first.split()[-1].lower()
    return ''
def main():
    with concurrent.futures.ThreadPoolExecutor(3) as pool:results=list(pool.map(fetcher.fetch,QUERIES))
    (HERE/'search_results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
    merged=[]
    for result in results:
        print(result['source'],result['status'],result.get('error',''),len(result.get('papers',[])),flush=True)
        for paper in result.get('papers',[]):
            doi=(paper.get('doi') or '').lower().removeprefix('https://doi.org/').strip();found=None
            for old in merged:
                d=(old.get('doi') or '').lower().removeprefix('https://doi.org/').strip()
                same=bool(doi and d and doi==d)
                if not (doi and d):
                    a,b=set(norm_title(paper['title']).split()),set(norm_title(old['title']).split())
                    same=bool(surname(paper) and surname(paper)==surname(old) and len(a&b)/max(1,len(a|b))>=.9)
                if same:found=old;break
            if found is not None:
                found['sources'].append(result['source'])
                for k,value in paper.items():
                    if value and not found.get(k):found[k]=value
            else:merged.append(dict(paper,sources=[result['source']]))
    (HERE/'deduplicated.json').write_text(json.dumps(merged,indent=2,ensure_ascii=False))
    for p in merged:print(p.get('date'),p.get('doi') or p.get('id'),p['title'],flush=True)
    (HERE/'search_manifest.json').write_text(json.dumps(dict(retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        academic_mcp_available=False,workflow='nature-academic-search/multi-source-search',
        raw_records=sum(len(r.get('papers',[])) for r in results),deduplicated=len(merged),
        discovery_only=True,source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.rglob('*') if p.is_file() and p.suffix in {'.py','.json','.txt'}}),indent=2))

if __name__=='__main__':main()
