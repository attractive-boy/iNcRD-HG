"""Independent streamed digest recheck of complete archive and frozen dependencies."""
import json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 m=json.loads((HERE/'archive_manifest.json').read_text());checks=[]
 for p,h in m['artifact_hashes'].items():assert sha(HERE/p)==h,p;checks.append(p)
 for p,h in m['frozen_dependencies'].items():assert sha(p)==h,p;checks.append(p)
 r=dict(status='passed',hash_checks=len(checks),archive_sha256=sha(HERE/'archive_manifest.json'),selection_sha256=sha(HERE/'selection.json'))
 (HERE/'archive_recheck.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
