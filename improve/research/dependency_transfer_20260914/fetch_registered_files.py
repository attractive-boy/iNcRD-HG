"""Download public DepMap files explicitly registered in Bioconductor's catalog.

The package catalog provides exact public URLs, sizes and MD5 values; no
authentication, portal challenge, guessed file identifier or private API is used.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import pandas as pd
import requests


ROOT=Path(__file__).resolve().parent
DEST=ROOT.parents[1]/'results'/'dependency_data_20260914'/'raw'


def fetch(name):
    catalog=ROOT/'bioconductor_dmfiles.csv'
    frame=pd.read_csv(catalog)
    rows=frame[(frame.dataset_id==25880521)&(frame.name==name)]
    if len(rows)!=1:raise ValueError('Require one file in the frozen public catalog')
    row=rows.iloc[0]
    DEST.mkdir(parents=True,exist_ok=True)
    path=DEST/name;manifest=path.with_suffix(path.suffix+'.json')
    if path.exists():
        if hashlib.md5(path.read_bytes()).hexdigest()!=row.md5:raise ValueError('Existing file checksum differs')
        print('EXISTS',path,flush=True);return
    record=dict(source_url=row.download_url,filename=name,release='DepMap 24Q2 Public',
                dataset_id=25880521,expected_md5=row.md5,expected_bytes=int(row['size']),
                catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
                retrieved_at_utc=datetime.now(timezone.utc).isoformat(),complete=False)
    try:
        with requests.get(row.download_url,stream=True,timeout=(20,40)) as response:
            record['http_status']=response.status_code
            record['content_type']=response.headers.get('Content-Type')
            response.raise_for_status()
            md5=hashlib.md5();sha=hashlib.sha256();size=0;checkpoint=0
            start=time.perf_counter();partial=path.with_suffix(path.suffix+'.part')
            with partial.open('wb') as output:
                for block in response.iter_content(1024*1024):
                    if not block:continue
                    if size==0 and block.lstrip().lower().startswith((b'<html',b'<!doctype')):
                        raise ValueError('Received HTML instead of the registered CSV')
                    output.write(block);md5.update(block);sha.update(block);size+=len(block)
                    if size-checkpoint>=50*1024*1024:
                        print('DOWNLOADED',name,size,round(time.perf_counter()-start,1),flush=True);checkpoint=size
            record.update(bytes=size,md5=md5.hexdigest(),sha256=sha.hexdigest())
            if size!=int(row['size']) or md5.hexdigest()!=row.md5:raise ValueError('Public catalog size/checksum mismatch')
            partial.replace(path);record['complete']=True
    except Exception as error:
        record['error']=type(error).__name__+': '+str(error)
        manifest.write_text(json.dumps(record,indent=2)+'\n')
        raise
    manifest.write_text(json.dumps(record,indent=2)+'\n')
    print('VERIFIED',name,size,sha.hexdigest(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name',choices=('CRISPRGeneDependency.csv','Model.csv'))
    fetch(parser.parse_args().name)
