"""Fetch selected public model assets at pinned Git commits, verifying Git blobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request


ROOT=Path('/data1/licheng/incrd-hg/data_external/functional_priority_20260913/rna_models')
REPOS={'mirbind':('BioGeMT/miRBind_2.0','49a8f0c52984800a434d4e63a4a52dde2356a4e9'),
       'parnet':('mhorlacher/parnet','2f0570cf47d8bb927415a71f853a97eb7e94005c'),
       'parnet_compat':('mhorlacher/parnet','ea68bd62861a0658fa6c1a81c8c4f8736e2ed2ed')}
WEIGHTS={'mirbind':['models/pairwise_onehot_model_20260105_200141.pt','models/gene_level_model_20260113_164412.pt'],
         'parnet':['models/NewRBPNet_7M_Penalty-0.0_20250107.pt']}


def blob_hash(raw):
    return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def fetch_file(directory,repo,commit,item):
    name=item['path'];path=directory/name
    url=f'https://raw.githubusercontent.com/{repo}/{commit}/{name}'
    if path.exists():
        raw=path.read_bytes()
        if blob_hash(raw)!=item['sha']:
            raise ValueError('Cached Git blob mismatch: '+name)
    else:
        for attempt in range(3):
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'NcRNAFunctionalPilot/1.0'})
                with urllib.request.urlopen(req,timeout=25) as response:
                    raw=response.read(64*2**20)
                if len(raw)!=item['size'] or blob_hash(raw)!=item['sha']:
                    raise ValueError('Downloaded Git blob mismatch: '+name)
                path.parent.mkdir(parents=True,exist_ok=True)
                temp=path.with_suffix(path.suffix+'.partial');temp.write_bytes(raw);temp.replace(path)
                break
            except Exception:
                if attempt==2:
                    raise
                time.sleep(attempt+1)
    return dict(path=name,url=url,git_blob=item['sha'],sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',choices=REPOS)
    parser.add_argument('--weights',action='store_true')
    args=parser.parse_args();repo,commit=REPOS[args.model]
    directory=ROOT/repo.replace('/','__')
    if args.model=='parnet_compat':
        directory=directory/('compat_'+commit[:12])
    if not (directory/'discovery.json').exists():
        url=f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1'
        req=urllib.request.Request(url,headers={'User-Agent':'NcRNAFunctionalPilot/1.0'})
        with urllib.request.urlopen(req,timeout=25) as response:
            tree=json.load(response)
        directory.mkdir(parents=True,exist_ok=True)
        (directory/'discovery.json').write_text(json.dumps(dict(repository=repo,commit=commit,tree=tree),indent=2))
    discovery=json.loads((directory/'discovery.json').read_text())
    if discovery['commit']!=commit or discovery['tree'].get('truncated'):
        raise ValueError('Unverified or incomplete Git tree')
    def wanted(item):
        name=item['path']
        if item['type']!='blob':
            return False
        if args.weights:
            return name in WEIGHTS['parnet' if args.model=='parnet_compat' else args.model] or name.startswith('examples/masks/') or name=='parnet/assets/ENCODE.idx2symbol-cell.pt'
        if args.model.startswith('parnet'):
            return (name.startswith('parnet/') and name.endswith('.py')) or name in ('README.md','pyproject.toml','setup.py','requirements.txt','LICENSE','LICENSE.txt','examples/masks/README.md')
        return ((name.startswith('code/pairwise_binding_site_model/') or name.startswith('code/gene_level_model/shared/'))
                and name.endswith(('.py','.md'))) or name in ('README.md','LICENSE','LICENSE.txt','analysis/gene_level_model/README.md')
    items=[item for item in discovery['tree']['tree'] if wanted(item)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        records=list(pool.map(lambda item:fetch_file(directory,repo,commit,item),items))
    manifest=dict(repository=repo,commit=commit,retrieved_at_utc=datetime.now(timezone.utc).isoformat(),files=records)
    name='WEIGHT_FILES.json' if args.weights else 'SOURCE_FILES.json'
    (directory/name).write_text(json.dumps(manifest,indent=2)+'\n')
    print(repo,commit,'verified_files',len(records),'bytes',sum(r['bytes'] for r in records),flush=True)


if __name__=='__main__':
    main()
