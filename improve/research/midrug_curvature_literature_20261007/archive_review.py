"""Freeze primary-paper provenance and the explicitly limited reading scope."""
import json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((HERE/'search_manifest.json').read_text())
for p,h in manifest['hashes'].items():assert sha(HERE/p)==h
fetch=json.loads((HERE/'primary_fetch.json').read_text())
for r in fetch:assert r['status']=='ok' and sha(HERE/'raw'/f"{r['name']}.html")==r['sha256']
files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['review_manifest.json','review_recheck.json']]
out=dict(status='archived',discovery_records=47,deduplicated_records=47,primary_documents_retrieved=3,
    reading_scope='Partial primary-method/results reading; exact sections and limitations in READING_NOTES.md; author code and appendix proofs not audited',
    academic_mcp_available=False,new_training_data_downloads=0,new_pretrained_weight_downloads=0,
    artifact_hashes={str(p.relative_to(HERE)):sha(p) for p in sorted(files)})
p=HERE/'review_manifest.json';assert not p.exists();p.write_text(json.dumps(out,indent=2))
for n,h in out['artifact_hashes'].items():assert sha(HERE/n)==h
(HERE/'review_recheck.json').write_text(json.dumps(dict(status='passed',hash_checks=len(out['artifact_hashes']),review_manifest_sha256=sha(p)),indent=2))
print(sha(p),len(out['artifact_hashes']))
