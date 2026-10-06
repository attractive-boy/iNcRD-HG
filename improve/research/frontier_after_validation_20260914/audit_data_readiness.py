"""Read feature-cache headers and search filenames, without outcome arrays."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
WORKSPACE = REPO.parent


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def main():
    output = ROOT / 'DATA_READINESS_AUDIT.json'
    if output.exists():
        raise ValueError('Refuse overwrite')
    pattern = re.compile(r'CRISPRGeneEffect|CRISPRGeneDependency|Chronos|PRISM|LINCS|L1000|OmicsExpression|Model\.csv', re.I)
    searches = []
    for directory in [WORKSPACE / 'data_external', WORKSPACE / '.external', REPO / '.external']:
        result = dict(directory=str(directory), exists=directory.is_dir(), matches=[])
        if directory.is_dir():
            proc = subprocess.run(['rg', '--files', '-uu', str(directory)], text=True, capture_output=True, check=True)
            result['matches'] = sorted(p for p in proc.stdout.splitlines() if pattern.search(Path(p).name))
            result['files_listed'] = len(proc.stdout.splitlines())
        searches.append(result)
    caches = []
    for encoder in ['mirbind', 'parnet']:
        path = REPO / 'improve/results/rna_functional_inputs_20260914_v2' / encoder / 'representations.npz'
        headers = {}
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                with archive.open(name) as handle:
                    version = np.lib.format.read_magic(handle)
                    if version == (1, 0):
                        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
                    elif version == (2, 0):
                        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
                    else:
                        raise ValueError('Unexpected NPY header version')
                    headers[name.removesuffix('.npy')] = dict(shape=list(shape), dtype=str(dtype), fortran_order=fortran_order)
        caches.append(dict(encoder=encoder, path=str(path), sha256=digest(path), array_headers=headers,
                           array_values_loaded=False))
    result = dict(as_of='2026-09-14', completed_at_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=digest(Path(__file__)), searches=searches, feature_caches=caches,
        association_or_source_outcome_labels_read=False,
        limitation='Filename search in listed roots is not proof that data do not exist elsewhere or under unrelated filenames. Header names establish cached fields, not correctness of model predictions.')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(searches=searches, feature_caches=[dict(encoder=c['encoder'], headers=c['array_headers']) for c in caches]), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
