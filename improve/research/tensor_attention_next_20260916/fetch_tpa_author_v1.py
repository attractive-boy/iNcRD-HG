"""Pin public TPA paper and author tree without running author code."""
from concurrent.futures import ThreadPoolExecutor
import json, urllib.parse
from pathlib import Path
from improve.research.tensor_attention_next_20260916.fetch_tpa_sources_v1 import retrieve
HERE = Path(__file__).resolve().parent

def main():
    dest = HERE / 'TPA_AUTHOR_DISCOVERY_V1.json'
    if dest.exists():
        raise ValueError('Preserve retrieval')
    jobs = [
        ('tpa_v7.pdf', 'https://arxiv.org/pdf/2501.06425v7'),
        ('tpa_author_repo_v1.json', 'https://api.github.com/repos/tensorgi/TPA'),
        ('tpa_crossref_exact_v1.json', 'https://api.crossref.org/works/10.52202/085713-3748'),
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(retrieve, jobs))
    meta = json.loads((HERE/'tpa_author_repo_v1.json').read_text())
    branch = urllib.parse.quote(meta['default_branch'], safe='')
    rows.append(retrieve(('tpa_author_commit_v1.json', 'https://api.github.com/repos/tensorgi/TPA/commits/'+branch)))
    sha = json.loads((HERE/'tpa_author_commit_v1.json').read_text())['sha']
    assert len(sha)==40 and all(c in '0123456789abcdef' for c in sha)
    rows.append(retrieve(('tpa_author_tree_v1.json', 'https://api.github.com/repos/tensorgi/TPA/git/trees/'+sha+'?recursive=1')))
    dest.write_text(json.dumps(dict(commit=sha, records=rows),indent=2)+'\n')
    for r in rows: print(json.dumps(r),flush=True)
    tree=json.loads((HERE/'tpa_author_tree_v1.json').read_text())
    for r in tree.get('tree',[]):
        if r['type']=='blob' and (r['path'].endswith('.py') or 'readme' in r['path'].lower()):
            print('AUTHOR_FILE',r['path'],r['sha'],flush=True)

if __name__=='__main__': main()
