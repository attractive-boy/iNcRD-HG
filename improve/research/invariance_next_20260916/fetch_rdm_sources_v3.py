"""One network retry and the official open-access conference landing page."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from fetch_rdm_sources_v2 import fetch

HERE = Path(__file__).resolve().parent
OUT = HERE / 'RDM_SOURCE_FETCH_V3.json'
if OUT.exists():
    raise ValueError('Preserve source manifest')
jobs = [
    ('rdm_arxiv_2310.18598v1_retry.pdf', 'https://arxiv.org/pdf/2310.18598v1', None),
    ('rdm_cvf_landing_v1.html', 'https://openaccess.thecvf.com/content/WACV2024/html/Nguyen_Domain_Generalisation_via_Risk_Distribution_Matching_WACV_2024_paper.html', None),
]
with ThreadPoolExecutor(max_workers=2) as pool:
    records = list(pool.map(fetch, jobs))
data = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            helper_sha256=hashlib.sha256((HERE/'fetch_rdm_sources_v2.py').read_bytes()).hexdigest(),
            reason='Initial arXiv PDF request reset before receiving bytes; one retry. CVF URL is a candidate until verified.',
            records=records)
OUT.write_text(json.dumps(data, indent=2) + '\n')
print(json.dumps(data, indent=2), flush=True)
