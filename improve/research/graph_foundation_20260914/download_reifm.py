"""Fetch the selected public checkpoint at an immutable revision, verify LFS."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent.parent/'results'/'reifm_assets_20260914'
REVISION = '2faf7472fd14c27ab1014b44d1df23c5e12877f8'
ENDPOINT = 'https://hf-mirror.com'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    logs = []

    def get(url, destination):
        rec = dict(url=url, destination=str(destination), retrieved_at_utc=datetime.now(timezone.utc).isoformat())
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'NcRDGraphResearch/1.0'})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                rec.update(status=r.status, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_bytes() != raw:
                raise ValueError('Existing artifact differs')
            destination.write_bytes(raw)
            logs.append(rec)
            print(rec, flush=True)
            return raw
        except Exception as exc:
            rec['error'] = repr(exc)
            logs.append(rec)
            raise

    try:
        url = f'{ENDPOINT}/api/models/itmatr/REIFM/tree/{REVISION}/gat_seed0'
        tree = json.loads(get(url, OUT/'gat_seed0_tree.json'))
        for name in ('best.pt', 'results.json'):
            row = next(x for x in tree if x['path'] == 'gat_seed0/'+name)
            data = get(f'{ENDPOINT}/itmatr/REIFM/resolve/{REVISION}/gat_seed0/{name}', OUT/'gat_seed0'/name)
            if len(data) != row['size']:
                raise ValueError('Size mismatch')
            if row.get('lfs') and hashlib.sha256(data).hexdigest() != row['lfs']['oid']:
                raise ValueError('LFS SHA-256 mismatch')
        get(f'{ENDPOINT}/itmatr/REIFM/resolve/{REVISION}/README.md', OUT/'README.md')
        (OUT/'VERIFIED.json').write_text(json.dumps(dict(
            model='itmatr/REIFM', revision=REVISION, selected='gat_seed0',
            checkpoint_sha256=hashlib.sha256((OUT/'gat_seed0'/'best.pt').read_bytes()).hexdigest(),
            checkpoint_size_and_lfs_hash_verified=True,
            public_mirror_reason='Official Hugging Face endpoint unreachable; public ungated model, immutable revision and repository LFS hash checked.',
            files={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file()}), indent=2)+'\n')
    finally:
        (ROOT/'reifm_checkpoint_requests.json').write_text(json.dumps(logs, indent=2)+'\n')


if __name__ == '__main__':
    main()
