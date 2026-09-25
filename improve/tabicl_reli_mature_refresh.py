"""Resolve historical miRNA names through the official miRBase 22 snapshot.

No arm is inferred. An exact historical name must establish a MIMAT accession.
Run before encoding MiDrug; already encoded lncRNA and drug files are unchanged.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from improve.tabicl_reli_inputs import DEFAULT_OUT, RESEARCH, digest, mature_index, write_json


def refresh(out):
    out = Path(out)
    if (out/'rna_MiDrug.npz').exists():
        raise ValueError('Cannot change an already encoded miRNA input')
    historical_path = RESEARCH/'mirbase_22_mature.fa'
    historical, _ = mature_index(historical_path)
    current, accessions = mature_index(RESEARCH/'mirbase_mature.fa')
    rows = json.loads((out/'MiDrug_sequences.json').read_text())
    manifest = json.loads((out/'inputs.json').read_text())
    if digest(out/'MiDrug_sequences.json') != manifest['files']['MiDrug_sequences.json']:
        raise ValueError('Previous mapping changed unexpectedly')
    history = out/'before_release22'
    if history.exists():
        raise ValueError('Historical refresh already performed')
    history.mkdir()
    for name in ['MiDrug_sequences.json', 'MiDrug_sequences.fasta', 'inputs.json']:
        (history/name).write_bytes((out/name).read_bytes())
    for row in rows:
        if row['status'] == 'current_exact':
            continue
        legacy = historical.get(row['name'].casefold(), [])
        if len(legacy) != 1:
            continue
        matches = accessions.get(legacy[0]['accession'], [])
        if matches and len({m['sequence'] for m in matches}) == 1:
            match = sorted(matches, key=lambda x: x['source_name'])[0]
            row.update(match, status='historical_exact_accession_current_sequence',
                       historical_source_name=legacy[0]['source_name'], historical_release='22',
                       historical_sequence_changed=legacy[0]['sequence'] != match['sequence'])
        elif not matches:
            row.update(legacy[0], status='historical_exact_absent_current', historical_release='22')
    write_json(out/'MiDrug_sequences.json', rows)
    (out/'MiDrug_sequences.fasta').write_text(''.join(
        '>'+r['name']+'\n'+r['sequence']+'\n' for r in rows if r['sequence']))
    for name in ['MiDrug_sequences.json', 'MiDrug_sequences.fasta']:
        manifest['files'][name] = digest(out/name)
    for path in [historical_path, Path(__file__)]:
        manifest['source_sha256'][str(path.resolve())] = digest(path)
    manifest['tasks']['MiDrug'] = dict(nodes=len(rows), resolved=sum(bool(r['sequence']) for r in rows),
        statuses=dict(Counter(r['status'] for r in rows)), max_length=max(len(r['sequence']) for r in rows))
    manifest['historical_refresh'] = dict(release='22', prior_manifest_sha256=digest(history/'inputs.json'),
        official_fetch_record=str(RESEARCH/'mirbase_22_fetch.json'),
        unresolved=[r['name'] for r in rows if not r['sequence']])
    write_json(out/'inputs.json', manifest)
    print(json.dumps(manifest['tasks']['MiDrug'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default=str(DEFAULT_OUT))
    args = parser.parse_args()
    refresh(args.out_dir)
