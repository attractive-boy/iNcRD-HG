"""Inspect only remote Parquet footers using validated public HTTP ranges."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / 'PROFILE_PARQUET_LAYOUT_V1.json'
if OUT.exists():
    raise ValueError('Preserve layout probe')
probes = json.loads((HERE / 'PROFILE_AND_METHOD_PROBE_V1.json').read_text())['records']


class Ranges(io.RawIOBase):
    def __init__(self, row):
        self.url = row['url']
        self.size = int(row['headers']['Content-Length'])
        self.etag = row['headers']['ETag']
        self.position = 0
        self.requests = []
        self.session = requests.Session()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0 or self.position > self.size:
            raise ValueError('Invalid range seek')
        return self.position

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        if count == 0:
            return b''
        if count > 64 * 1024 * 1024:
            raise ValueError('Metadata-only probe requested too many bytes')
        start, end = self.position, self.position + count - 1
        response = self.session.get(self.url, headers={'Range': 'bytes=%d-%d' % (start, end), 'If-Match': self.etag}, timeout=45)
        if (response.status_code != 206 or response.headers.get('Content-Range') != 'bytes %d-%d/%d' % (start, end, self.size)
                or response.headers.get('ETag') != self.etag or len(response.content) != count):
            raise ValueError('Unverified HTTP range response: %s %s' % (response.status_code, dict(response.headers)))
        self.requests.append(dict(start=start, end=end, bytes=count, sha256=hashlib.sha256(response.content).hexdigest()))
        self.position += count
        return response.content


records = []
for source in probes:
    if source['method'] != 'HEAD':
        continue
    row = dict(name=source['name'], url=source['url'], etag=source['headers']['ETag'], bytes=int(source['headers']['Content-Length']))
    reader = Ranges(source)
    try:
        metadata = pq.ParquetFile(reader)
        row.update(rows=metadata.metadata.num_rows, columns=metadata.schema_arrow.names,
            schema=str(metadata.schema_arrow), row_groups=metadata.metadata.num_row_groups,
            groups=[dict(rows=metadata.metadata.row_group(i).num_rows, bytes=metadata.metadata.row_group(i).total_byte_size)
                    for i in range(metadata.metadata.num_row_groups)])
    except Exception as exc:
        row['error'] = str(exc)
    row['range_requests'] = reader.requests
    records.append(row)
    print(row['name'], row.get('rows'), row.get('row_groups'), row.get('error', ''), flush=True)
OUT.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), pyarrow_version=__import__('pyarrow').__version__,
    records=records, association_outcomes_read=False, profile_measurements_read=False), indent=2) + '\n')
