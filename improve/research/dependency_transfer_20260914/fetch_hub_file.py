"""Fetch an explicitly registered public ExperimentHub derivative."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urljoin

import requests

ROOT=Path(__file__).resolve().parent
DEST=ROOT.parents[1]/'results'/'dependency_data_20260914'/'raw'


def fetch(title):
    catalog=ROOT/('experimenthub_drug_record.json' if title.startswith('drug_sensitivity') else 'experimenthub_selected_records.json')
    rows=[r for r in json.loads(catalog.read_text()) if r['title']==title]
    if len(rows)!=1:raise ValueError('Expected one registered resource')
    row=rows[0];url=urljoin(row['location_prefix'],row['rdatapath'])
    dest=DEST/Path(row['rdatapath']).name;DEST.mkdir(parents=True,exist_ok=True)
    manifest=dest.with_suffix(dest.suffix+'.json')
    if dest.exists():
        m=json.loads(manifest.read_text())
        if not m['complete'] or hashlib.sha256(dest.read_bytes()).hexdigest()!=m['sha256']:
            raise ValueError('Existing derivative differs')
        print('EXISTS',dest,flush=True);return
    record=dict(resource=row,url=url,catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
                retrieved_at_utc=datetime.now(timezone.utc).isoformat(),complete=False,
                semantics=('PRISM 19Q3 primary-screen logfold viability relative to DMSO; the 21Q2 package update adds compound metadata, not new experiments or IC50 values.' if title.startswith('drug_sensitivity') else '22Q2 gene-effect derivative; not the 24Q4/25Q2 CRISPRGeneDependency probability file.'))
    try:
        with requests.get(url,stream=True,timeout=(20,40)) as response:
            record.update(http_status=response.status_code,content_type=response.headers.get('Content-Type'),
                          etag=response.headers.get('ETag'),content_length=response.headers.get('Content-Length'))
            response.raise_for_status()
            sha=hashlib.sha256();md5=hashlib.md5();size=0;last=0;start=time.perf_counter()
            partial=dest.with_suffix(dest.suffix+'.part')
            with partial.open('wb') as stream:
                for block in response.iter_content(1024*1024):
                    if not block:continue
                    if size==0 and block.lstrip().lower().startswith((b'<html',b'<!doctype')):
                        raise ValueError('HTML response instead of R data')
                    stream.write(block);sha.update(block);md5.update(block);size+=len(block)
                    if size-last>=25*1024*1024:
                        print('DOWNLOADED',title,size,round(time.perf_counter()-start,1),flush=True);last=size
            record.update(bytes=size,sha256=sha.hexdigest(),md5=md5.hexdigest())
            if record['content_length'] and size!=int(record['content_length']):raise ValueError('Truncated response')
            etag=(record['etag'] or '').strip('"')
            record['etag_md5_checked']=bool(re.fullmatch('[0-9a-fA-F]{32}',etag))
            if record['etag_md5_checked'] and md5.hexdigest()!=etag.lower():raise ValueError('ETag/MD5 mismatch')
            partial.replace(dest);record['complete']=True
    except Exception as error:
        record['error']=type(error).__name__+': '+str(error)
        manifest.write_text(json.dumps(record,indent=2)+'\n');raise
    manifest.write_text(json.dumps(record,indent=2)+'\n')
    print('VERIFIED',title,size,sha.hexdigest(),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('title',choices=('crispr_22Q2','metadata_22Q2','drug_sensitivity_21Q2'))
    fetch(ap.parse_args().title)
