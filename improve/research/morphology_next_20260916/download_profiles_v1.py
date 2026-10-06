"""Fetch two declared public feature tables in verified ranges; retain failures."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[3]
DEST = WORK / 'data_external/morphology_20260916'
OUT = HERE / 'PROFILE_DOWNLOAD_V1.json'
if OUT.exists() or DEST.exists():
    raise ValueError('Preserve prior download; a recovery must use a new declared version')
DEST.mkdir(parents=True)
probes = json.loads((HERE / 'PROFILE_AND_METHOD_PROBE_V1.json').read_text())['records']
sources = [row for row in probes if row['method'] == 'HEAD']
CHUNK = 32 * 1024 * 1024


def utc():
    return datetime.now(timezone.utc).isoformat()


def download_chunk(source, start, end):
    label = source['name']
    etag = source['headers']['ETag']
    size = int(source['headers']['Content-Length'])
    row = dict(name=label, start=start, end=end, attempts=[])
    for attempt in (1, 2):
        path = DEST / ('%s_%012d_%012d_attempt%d.part' % (label, start, end, attempt))
        record = dict(utc=utc(), attempt=attempt, file=str(path))
        try:
            response = requests.get(source['url'],
                headers={'Range': 'bytes=%d-%d' % (start, end), 'If-Match': etag}, timeout=(15, 40), stream=True)
            record['http_status'] = response.status_code
            if (response.status_code != 206 or response.headers.get('ETag') != etag
                    or response.headers.get('Content-Range') != 'bytes %d-%d/%d' % (start, end, size)):
                raise ValueError('Unexpected range or object version')
            h = hashlib.sha256()
            count = 0
            with path.open('xb') as stream:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    stream.write(block)
                    h.update(block)
                    count += len(block)
                    if count > end - start + 1:
                        raise ValueError('Range exceeds declared length')
            if count != end - start + 1:
                raise ValueError('Incomplete range')
            record.update(bytes=count, sha256=h.hexdigest(), complete=True)
            row['attempts'].append(record)
            row['selected_file'] = str(path)
            row['sha256'] = h.hexdigest()
            break
        except Exception as exc:
            record.update(error=str(exc), complete=False)
            row['attempts'].append(record)
    record_path = DEST / ('%s_%012d_%012d.json' % (label, start, end))
    record_path.write_text(json.dumps(row, indent=2) + '\n')
    print('PROFILE_CHUNK', label, start, end, 'OK' if 'selected_file' in row else 'FAILED', flush=True)
    return row


jobs = [(source, start, min(start + CHUNK - 1, int(source['headers']['Content-Length']) - 1))
        for source in sources for start in range(0, int(source['headers']['Content-Length']), CHUNK)]
records = []
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(download_chunk, *job) for job in jobs]
    for future in as_completed(futures):
        records.append(future.result())
files = []
for source in sources:
    chunks = sorted([r for r in records if r['name'] == source['name']], key=lambda row: row['start'])
    record = dict(name=source['name'], url=source['url'], etag=source['headers']['ETag'], chunks=chunks)
    if any('selected_file' not in row for row in chunks):
        record['error'] = 'Incomplete source ranges; no assembled input'
        files.append(record)
        continue
    path = DEST / (source['name'] + '_v1.parquet')
    sha, md5 = hashlib.sha256(), hashlib.md5()
    count = 0
    with path.open('xb') as stream:
        for row in chunks:
            digest = hashlib.sha256()
            with Path(row['selected_file']).open('rb') as part:
                for block in iter(lambda: part.read(1024 * 1024), b''):
                    stream.write(block)
                    sha.update(block)
                    md5.update(block)
                    digest.update(block)
                    count += len(block)
            if digest.hexdigest() != row['sha256']:
                raise ValueError('Downloaded part changed')
    if count != int(source['headers']['Content-Length']) or md5.hexdigest() != source['headers']['ETag'].strip('"'):
        raise ValueError('Assembled object length or MD5 does not match S3')
    record.update(file=str(path), bytes=count, sha256=sha.hexdigest(), md5=md5.hexdigest(), s3_etag_verified=True)
    files.append(record)
    print('PROFILE_ASSEMBLED', source['name'], count, sha.hexdigest(), flush=True)
result = dict(utc=utc(), files=files, all_complete=all('file' in row for row in files),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    declared_probe_sha256=hashlib.sha256((HERE / 'PROFILE_AND_METHOD_PROBE_V1.json').read_bytes()).hexdigest(),
    no_images_downloaded=True, association_outcomes_read=False, new_neural_fits=0)
OUT.write_text(json.dumps(result, indent=2) + '\n')
print('PROFILE_DOWNLOAD_COMPLETE', result['all_complete'], flush=True)
