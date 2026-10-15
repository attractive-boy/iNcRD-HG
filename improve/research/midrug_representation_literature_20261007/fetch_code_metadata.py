"""Pin public implementation metadata before reading selected source files."""
from fetch_primary import fetch, HERE
import concurrent.futures
import json

if __name__ == '__main__':
    jobs=[
        ('nodejepa_repo.json','https://api.github.com/repos/OliverZ-dot/Node-Jepa'),
        ('vicreg_repo.json','https://api.github.com/repos/facebookresearch/vicreg'),
        ('vicreg_meta.xml','https://export.arxiv.org/api/query?id_list=2105.04906'),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: rows=list(pool.map(fetch,jobs))
    (HERE/'code_metadata_fetch.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
