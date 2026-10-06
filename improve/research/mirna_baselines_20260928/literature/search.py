"""No-MCP fallback: save public OpenAlex discovery records, not inferred citations."""
import json,time,urllib.request,urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
QUERIES={
 'plmf':'A pre-trained language model-based cross-modal fusion framework for predicting miRNA-drug resistance and sensitivity associations',
 'dlmvf':'Attention-Guided Multiview Deep Learning Framework Uncovers miRNA-Drug Associations for Therapeutic Discovery',
 'dlstmda':'DLST-MDA', 'subknet':'SubKNet', 'gcnnmma':'GCNNMMA',
 'gcmdr':'GCMDR miRNA drug resistance', 'ammgc':'AMMGC miRNA',
 'lagcn':'layer attention graph convolutional miRNA drug resistance',
 'jmssmma':'JMSSMMA', 'graphdta':'GraphDTA', 'mldti':'ML-DTI mutual learning',
 'submdta':'SubMDTA',
}
def search(item):
    name,q=item; url='https://api.openalex.org/works?'+urllib.parse.urlencode({'search':q,'per-page':5})
    try:
        with urllib.request.urlopen(url,timeout=30) as r: raw=json.load(r)
        records=[]
        for w in raw.get('results',[]):
            source=(w.get('primary_location') or {}).get('source') or {}
            records.append(dict(title=w.get('title'),doi=w.get('doi'),year=w.get('publication_year'),
                                journal=source.get('display_name'),id=w['id'],authors=[a['author']['display_name'] for a in w.get('authorships',[])],
                                abstract_inverted_index=w.get('abstract_inverted_index'),cited_by_count=w.get('cited_by_count')))
        out={'query':q,'url':url,'records':records}
    except Exception as e:out={'query':q,'url':url,'error':str(e)}
    Path(__file__).with_name(name+'.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print(name,[(r['title'],r['doi']) for r in out.get('records',[])[:2]],flush=True)
if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(search,QUERIES.items()))
