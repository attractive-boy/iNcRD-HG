"""Label-free sequence and atom evidence for the shared RELI-TabICL probe.

Run prepare with the tabular environment, encode/drugs with ncRNAdrug.
Original entity files, checkpoints and association splits are never modified.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'incrd-run'
RESEARCH = Path(__file__).parent / 'research/mirna_general_architecture_20260912'
DEFAULT_OUT = Path(__file__).parent / 'results/tabicl_reli_inputs_20260912'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def fasta(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    header, chunks = None, []
    with opener(path, 'rt') as handle:
        for line in handle:
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(chunks).upper().replace('T', 'U')
                header, chunks = line[1:].strip(), []
            elif line.strip():
                if header is None:
                    raise ValueError('Sequence before FASTA header')
                chunks.append(line.strip())
    if header is not None:
        yield header, ''.join(chunks).upper().replace('T', 'U')


def node_names(task):
    filename = {'LncDrug': 'lncRNA.csv', 'MiDrug': 'miRNA.csv'}[task]
    with (DATA / '2.construct_network' / filename).open() as handle:
        names = [row['x'].strip() for row in csv.DictReader(handle)]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate node identifiers')
    return names


def mature_index(path):
    names, accessions = defaultdict(list), defaultdict(list)
    for header, sequence in fasta(path):
        parts = header.split()
        if not parts[0].casefold().startswith('hsa-'):
            continue
        row = dict(source_name=parts[0], accession=parts[1], sequence=sequence)
        names[parts[0].casefold()].append(row)
        accessions[parts[1]].append(row)
    return names, accessions


def prepare(out):
    out = Path(out)
    if (out / 'inputs.json').exists():
        raise ValueError('Inputs already prepared; use existing inputs or a new directory')
    out.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__)]
    current_path = RESEARCH / 'mirbase_mature.fa'
    historical_path = RESEARCH / 'mirbase_22_1_mature.fa'
    current, accessions = mature_index(current_path)
    sources.append(current_path)
    historical = {}
    if historical_path.exists():
        historical, _ = mature_index(historical_path)
        sources.append(historical_path)
    mirna = []
    for name in node_names('MiDrug'):
        exact = current.get(name.casefold(), [])
        row = dict(name=name, sequence='', status='unresolved', source_name='', accession='')
        if len(exact) == 1:
            row.update(exact[0], status='current_exact')
        else:
            legacy = historical.get(name.casefold(), [])
            if len(legacy) == 1:
                candidates = accessions.get(legacy[0]['accession'], [])
                if candidates and len({r['sequence'] for r in candidates}) == 1:
                    row.update(sorted(candidates, key=lambda x: x['source_name'])[0],
                               status='historical_exact_accession_current_sequence',
                               historical_source_name=legacy[0]['source_name'],
                               historical_sequence_changed=legacy[0]['sequence'] != candidates[0]['sequence'])
                elif not candidates:
                    row.update(legacy[0], status='historical_exact_absent_current')
        mirna.append(row)

    lnc_names = node_names('LncDrug')
    remaining, found = set(lnc_names), {}
    for filename in ['gencode.v44.lncRNA_transcripts.fa.gz', 'gencode.v33.lncRNA.fa.gz',
                     'gencode.v26.lncRNA.fa.gz', 'gencode.v22.lncRNA.fa.gz']:
        path = DATA / '11.seq_data/gencode' / filename
        sources.append(path)
        candidates = {}
        for header, sequence in fasta(path):
            parts = header.split('|')
            if len(parts) < 7 or parts[5] not in remaining:
                continue
            name, transcript = parts[5], parts[0]
            row = dict(name=name, sequence=sequence, status='annotated_full_transcript',
                       source_file=filename, transcript=transcript, gene_id=parts[1])
            previous = candidates.get(name)
            if previous is None or (-len(sequence), transcript) < (-len(previous['sequence']), previous['transcript']):
                candidates[name] = row
        found.update(candidates)
        remaining.difference_update(candidates)
    old_path = DATA / '11.seq_data/sequences/lncRNA_seq.fasta'
    sources.append(old_path)
    legacy = {h.split()[0]: s for h, s in fasta(old_path)}
    lncrna = []
    for name in lnc_names:
        row = found.get(name, dict(name=name, sequence=legacy.get(name, ''),
                                  status='legacy_partial_fallback' if name in legacy else 'unresolved'))
        if name in legacy and row['sequence']:
            row['legacy_is_prefix'] = row['sequence'].startswith(legacy[name])
        lncrna.append(row)
    for filename in ['miRNA.csv', 'lncRNA.csv', 'Drug.csv', 'drug_smiles.csv']:
        sources.append(DATA / '2.construct_network' / filename)
    for task, rows in [('MiDrug', mirna), ('LncDrug', lncrna)]:
        for row in rows:
            if set(row['sequence']) - set('ACGUN'):
                raise ValueError('Unexpected nucleotide alphabet: ' + row['name'])
        write_json(out / (task + '_sequences.json'), rows)
        (out / (task + '_sequences.fasta')).write_text(''.join(
            '>' + row['name'] + '\n' + row['sequence'] + '\n' for row in rows if row['sequence']))
    summary = dict(
        source_sha256={str(p.resolve()): digest(p) for p in sources},
        files={name: digest(out / name) for name in [
            'MiDrug_sequences.json', 'LncDrug_sequences.json', 'MiDrug_sequences.fasta', 'LncDrug_sequences.fasta']},
        tasks={task: dict(nodes=len(rows), resolved=sum(bool(r['sequence']) for r in rows),
                         statuses=dict(Counter(r['status'] for r in rows)),
                         max_length=max(len(r['sequence']) for r in rows))
               for task, rows in [('MiDrug', mirna), ('LncDrug', lncrna)]},
        selection='Exact current miRBase names; otherwise exact historical names mapped by MIMAT accession. No arm guessing. lncRNA exact gene symbol, most recent available GENCODE release, longest transcript with accession tie-break.',
        no_association_labels=True,
        limitations=['Historical-only miRNA accessions are retained with explicit status, not asserted to be current validated miRNAs.',
                     'Longest annotated lncRNA transcript is a gene-level representative; not sample-specific isoform inference.',
                     'Legacy partial sequence fallback and unresolved entities remain in all evaluations.'])
    write_json(out / 'inputs.json', summary)
    print(json.dumps(summary['tasks'], indent=2), flush=True)


def local_sites(length, width=8, stride=4, max_sites=64, context_length=510):
    """Return sites, coverage weights and the containing encoder window starts."""
    if length == 0:
        return [], np.empty(0, dtype=np.float32), []
    if length <= width:
        starts = [0]
    else:
        starts = sorted(set(list(range(0, length-width+1, stride)) + [length-width]))
    if len(starts) > max_sites:
        starts = [starts[i] for i in np.linspace(0, len(starts)-1, max_sites).round().astype(int)]
    sites = [(s, min(s+width, length)) for s in starts]
    coverage = Counter(pos for start, end in sites for pos in range(start, end))
    weights = np.array([sum(1/coverage[pos] for pos in range(start, end)) for start, end in sites], np.float32)
    weights /= weights.sum()
    windows = [max(0, min((start//448)*448, length-context_length)) for start, _ in sites]
    if any(not (window <= start < end <= min(window+context_length, length))
           for (start, end), window in zip(sites, windows)):
        raise ValueError('Site lies outside its encoder window')
    return sites, weights, windows


def encode(out, task, device, batch_size):
    import torch
    from multimolecule import RnaTokenizer, RnaFmModel
    from multimolecule.models.rnafm.configuration_rnafm import RnaFmConfig
    from safetensors import safe_open

    out = Path(out)
    target = out / ('rna_' + task + '.npz')
    if target.exists():
        raise ValueError('RNA evidence already exists')
    inputs = json.loads((out / 'inputs.json').read_text())
    source = out / (task + '_sequences.json')
    if digest(source) != inputs['files'][source.name]:
        raise ValueError('Prepared sequences changed')
    records = json.loads(source.read_text())
    checkpoint = DATA / 'models/rnafm'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20260912)
    model = RnaFmModel(RnaFmConfig.from_pretrained(str(checkpoint)))
    with safe_open(str(checkpoint / 'model.safetensors'), framework='pt') as handle:
        state = {k[len('model.'):]: handle.get_tensor(k) for k in handle.keys() if k.startswith('model.')}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if any(not name.startswith('pooler.') for name in missing) or unexpected:
        raise ValueError('Unloaded RNA encoder weights: ' + repr((missing, unexpected)))
    model = model.eval().requires_grad_(False).to(device)
    tokenizer = RnaTokenizer.from_pretrained(str(checkpoint))
    local = np.zeros((len(records), 64, 640), np.float32)
    weights = np.zeros((len(records), 64), np.float32)
    positions = np.full((len(records), 64, 2), -1, np.int32)
    windows = defaultdict(list)
    encoded_coverage = []
    for row_idx, record in enumerate(records):
        seq = record['sequence']
        sites, mass, starts = local_sites(len(seq))
        weights[row_idx, :len(sites)] = mass
        if sites:
            positions[row_idx, :len(sites)] = sites
        covered = set()
        for site_idx, ((start, end), window) in enumerate(zip(sites, starts)):
            key = (row_idx, window)
            windows[key].append((site_idx, start-window, end-window))
            covered.update(range(window, min(window+510, len(seq))))
        encoded_coverage.append(len(covered))
    keys = list(windows)
    for begin in range(0, len(keys), batch_size):
        batch_keys = keys[begin:begin+batch_size]
        sequences = [records[row]['sequence'][window:window+510] for row, window in batch_keys]
        encoded = tokenizer(sequences, return_tensors='pt', padding=True, truncation=False)
        for n, sequence in enumerate(sequences):
            if int(encoded['attention_mask'][n].sum()) != len(sequence)+2:
                raise ValueError('RNA tokenizer is not nucleotide-resolved')
        with torch.no_grad():
            hidden = model(input_ids=encoded['input_ids'].to(device),
                           attention_mask=encoded['attention_mask'].to(device)).last_hidden_state
        for batch_idx, key in enumerate(batch_keys):
            for site_idx, start, end in windows[key]:
                local[key[0], site_idx] = hidden[batch_idx, start+1:end+1].mean(0).float().cpu().numpy()
        if begin == 0 or (begin//batch_size) % 20 == 0:
            print('RNA_ENCODE', task, min(begin+batch_size, len(keys)), '/', len(keys), flush=True)
    if not np.isfinite(local).all():
        raise FloatingPointError('Nonfinite RNA embeddings')
    np.savez_compressed(target, local=local, weights=weights, positions=positions,
                        names=np.array([r['name'] for r in records]),
                        lengths=np.array([len(r['sequence']) for r in records], np.int32))
    write_json(out / ('rna_' + task + '.json'), dict(
        task=task, source_sha256=digest(source), encoder_sha256=digest(checkpoint/'model.safetensors'),
        configuration_sha256=digest(checkpoint/'config.json'), output_sha256=digest(target),
        implementation_sha256=digest(__file__), max_sites=64, width=8, stride=4, context_length=510,
        encoding='FP32 frozen RNA-FM, 8nt local means, at most 64 uniformly spaced sites, coverage-normalized site weights.',
        encoder_windows=len(keys), encoded_nucleotides=int(sum(encoded_coverage)),
        full_sequence_nucleotides=int(sum(len(r['sequence']) for r in records)),
        encoded_fraction_by_node=[covered/max(len(r['sequence']), 1) for covered, r in zip(encoded_coverage, records)],
        missing_unused_pooler_parameters=list(missing), no_association_labels=True))
    print('RNA_SAVED', task, target, flush=True)


def drugs(out):
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem
    out = Path(out)
    target = out / 'drug.npz'
    if target.exists():
        raise ValueError('Drug evidence already exists')
    drug_path = DATA/'2.construct_network/Drug.csv'
    smiles_path = DATA/'2.construct_network/drug_smiles.csv'
    with drug_path.open() as handle:
        records = list(csv.DictReader(handle))
    with smiles_path.open() as handle:
        smiles = {r['drug_cid']: r['smiles'] for r in csv.DictReader(handle)}
    local_rows, atom_counts = [], []
    for row in records:
        mol = Chem.MolFromSmiles(smiles[row['CID']])
        if mol is None:
            raise ValueError('Invalid drug SMILES: ' + row['Drug_Name'])
        mol = Chem.RemoveHs(mol)
        info = {}
        AllChem.GetMorganFingerprint(mol, 2, bitInfo=info)
        matrix = np.zeros((mol.GetNumAtoms(), 1024), np.float32)
        for identifier, sites in info.items():
            for atom, radius in sites:
                matrix[atom, identifier % 1024] += [0.5, 1.0, 1.0][radius]
        local_rows.append(matrix)
        atom_counts.append(len(matrix))
    local = np.zeros((len(records), max(atom_counts), 1024), np.float32)
    weights = np.zeros(local.shape[:2], np.float32)
    for i, matrix in enumerate(local_rows):
        local[i, :len(matrix)] = matrix
        weights[i, :len(matrix)] = 1/len(matrix)
    np.savez_compressed(target, local=local, weights=weights,
                        names=np.array([r['Drug_Name'] for r in records]))
    write_json(out/'drug.json', dict(
        output_sha256=digest(target), implementation_sha256=digest(__file__), rdkit_version=rdBase.rdkitVersion,
        source_sha256={str(p): digest(p) for p in [drug_path, smiles_path]},
        nodes=len(records), maximum_atoms=max(atom_counts),
        encoding='Atom-centered Morgan environments radii 0/1/2; weights 0.5/1/1; 1024 hashed dimensions. All molecular fragments retained; no learned weights.',
        no_association_labels=True))
    print('DRUG_SAVED', target, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'encode', 'drugs'])
    parser.add_argument('--out-dir', default=str(DEFAULT_OUT))
    parser.add_argument('--task', choices=['LncDrug', 'MiDrug'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=12)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out_dir)
    elif args.command == 'encode':
        if args.task is None:
            parser.error('--task is required for encode')
        encode(args.out_dir, args.task, args.device, args.batch_size)
    else:
        drugs(args.out_dir)


if __name__ == '__main__':
    main()
