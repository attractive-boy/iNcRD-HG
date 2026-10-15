"""Public metadata discovery only; no external model/data downloads."""
import concurrent.futures, json, urllib.request, urllib.parse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
queries={
 'epmc_midrug':'(TITLE_ABS:"miRNA" AND TITLE_ABS:"drug" AND TITLE_ABS:"prediction") AND FIRST_PDATE:[2025-01-01 TO 2026-10-07]',
 'epmc_expression':'(TITLE_ABS:"gene expression" AND TITLE_ABS:"normalization" AND TITLE_ABS:"prediction") AND FIRST_PDATE:[2025-01-01 TO 2026-10-07]',
 'crossref_midrug':'microRNA drug association prediction expression',
}
def one(item):
 key,q=item
 if key.startswith('epmc'):
  url='https://www.ebi.ac.uk/europepmc/webservices/rest/search?'+urllib.parse.urlencode(dict(query=q,format='json',pageSize=30,resultType='core',sort='P_PDATE_D desc'))
 else:
  url='https://api.crossref.org/works?'+urllib.parse.urlencode({'query':q,'filter':'from-pub-date:2025-01-01,until-pub-date:2026-10-07','rows':20,'sort':'relevance'})
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'miRNA-method-review/1.0'}),timeout=35) as r: raw=r.read()
  (ROOT/(key+'.json')).write_bytes(raw);obj=json.loads(raw)
  if key.startswith('epmc'):
   records=[{k:x.get(k) for k in ['title','authorString','doi','id','pmcid','firstPublicationDate','abstractText']} for x in obj['resultList']['result']]
  else:records=[dict(title=x.get('title'),doi=x.get('DOI'),date=x.get('published'),abstract=x.get('abstract')) for x in obj['message']['items']]
  return dict(query=q,url=url,status='ok',records=records)
 except Exception as e:return dict(query=q,url=url,status='failed',error=repr(e))
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(3) as ex: results=dict(zip(queries,ex.map(one,queries.items())))
 (ROOT/'search_results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
 for k,v in results.items():
  print(k,v['status'])
  for x in v.get('records',[]):print(x['doi'],x['title'])
