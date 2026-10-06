"""Fetch two public, ungated checkpoints; verify the Hub LFS SHA256."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path('/data1/licheng/incrd-hg/data_external/frontier_models_20260914')
SPECS = [
    dict(id='tabdpt1_3', repo='Layer6/TabDPT', revision='a5ca6e01c0fa09ec68c73e958e5199d1932abb3a',
         file='tabdpt1_3.safetensors', bytes=252233296,
         sha256='97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb'),
    dict(id='mitra2', repo='autogluon/mitra-classifier-2', revision='edada0d20759c58ada8c8605c25f22f6e98ea5f0',
         file='model.safetensors', bytes=302717904,
         sha256='5ffab0e2cf52f61c5b7c7eb1e8542996736d1023a0212190cc09abc2119a1e09'),
]


def fetch(spec):
    dest = ROOT / spec['id']
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / spec['file']
    url = 'https://hf-mirror.com/' + spec['repo'] + '/resolve/' + spec['revision'] + '/' + spec['file']
    record = dict(**spec, url=url, origin='https://huggingface.co/' + spec['repo'],
                  retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
                  mirror_reason='huggingface.co DNS resolution unavailable; public ungated model, pinned commit and Hub LFS hash')
    try:
        if not path.exists():
            part = path.with_suffix(path.suffix + '.part')
            req = urllib.request.Request(url, headers={'User-Agent': 'NcRDResearch/1.0'})
            with urllib.request.urlopen(req, timeout=45) as response, part.open('wb') as stream:
                while data := response.read(8 * 1024 * 1024):
                    stream.write(data)
            part.rename(path)
        h = hashlib.sha256()
        with path.open('rb') as stream:
            while data := stream.read(8 * 1024 * 1024):
                h.update(data)
        if path.stat().st_size != spec['bytes'] or h.hexdigest() != spec['sha256']:
            raise ValueError('Size or SHA256 mismatch')
        record.update(verified=True, path=str(path))
        if spec['id'] == 'mitra2':
            config = 'https://hf-mirror.com/' + spec['repo'] + '/resolve/' + spec['revision'] + '/config.json'
            with urllib.request.urlopen(config, timeout=25) as response:
                raw = response.read()
            json.loads(raw)
            (dest / 'config.json').write_bytes(raw)
            record['config_sha256'] = hashlib.sha256(raw).hexdigest()
    except Exception as exc:
        record.update(verified=False, error=repr(exc))
    (dest / 'DOWNLOAD.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(fetch, spec) for spec in SPECS]):
            print(json.dumps(future.result()), flush=True)
