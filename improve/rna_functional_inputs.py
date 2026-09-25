"""Label-free, matched-input RNA representations for the 2026-09-14 experiment.

Existing pilot artifacts are immutable. New downloads, frozen model outputs and
their provenance are written below a separate output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
from pathlib import Path
import time
import urllib.parse
import urllib.request

import numpy as np
from scipy import sparse

from improve import tabular_foundation as base
from improve import rna_site_windows as sites


DEFAULT_OUT = Path('improve/results/rna_functional_inputs_20260914_v2')
RNA = sites.RNA


def requested_accession(returned, requested):
    """Versioned requests require the exact version; bare IDs allow versions."""
    return any(returned == value if '.' in value else returned.split('.')[0] == value for value in requested)


def resolve_accession(requested, transcripts):
    if '.' in requested:
        return requested if requested in transcripts else None
    matches = [name for name in transcripts if name.split('.')[0] == requested]
    if len(matches) > 1:
        raise ValueError('Multiple returned versions for an unversioned accession')
    return matches[0] if matches else None


def verified_sequences(task):
    manifest = base.read_json(RNA / 'inputs.json')
    name = task + '_sequences.json'
    if base.digest(RNA / name) != manifest['files'][name]:
        raise ValueError('Frozen RNA sequence source changed')
    return base.read_json(RNA / name)


def verified_rnafm(task):
    meta = base.read_json(RNA / ('rna_' + task + '.json'))
    source, target = RNA / (task + '_sequences.json'), RNA / ('rna_' + task + '.npz')
    if base.digest(source) != meta['source_sha256'] or base.digest(target) != meta['output_sha256']:
        raise ValueError('RNA-FM input or representation changed')
    with np.load(target, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    records = verified_sequences(task)
    np.testing.assert_array_equal(data['names'], [r['name'] for r in records])
    np.testing.assert_array_equal(data['lengths'], [len(r['sequence']) for r in records])
    return data


def begin(directory, spec):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'VALIDATION.json').exists():
        raise ValueError('Completed artifact is immutable: ' + str(directory))
    p = base.seal(dict(spec, implementation_sha256=base.digest(Path(__file__))))
    if (directory / 'protocol.json').exists() and base.read_json(directory / 'protocol.json') != p:
        raise ValueError('Protocol changed; use a new directory')
    base.write_json(directory / 'protocol.json', p)
    return p


def finish(directory, protocol, files, **details):
    result = dict(protocol_sha256=protocol['sha256'], passed=True,
                  files={f: base.digest(directory / f) for f in files}, **details)
    base.write_json(directory / 'VALIDATION.json', result)
    print('VERIFIED_INPUT', directory, details, flush=True)


def verify(directory):
    p = base.check_seal(base.read_json(directory / 'protocol.json'))
    v = base.read_json(directory / 'VALIDATION.json')
    if not v['passed'] or v['protocol_sha256'] != p['sha256']:
        raise ValueError('Unverified representation')
    for f, digest in v['files'].items():
        if base.digest(directory / f) != digest:
            raise ValueError('Representation changed: ' + f)
    return v


def windows(args):
    """Keep every eligible drug-target site and every PMID supporting that site."""
    out = Path(args.out_dir) / 'windows'
    rna = {r['name']: dict(r, index=i) for i, r in enumerate(verified_sequences('MiDrug'))}
    dictionary = base.read_json(sites.REFERENCE / 'dictionary.json')
    reference = base.read_json(sites.REFERENCE / 'INPUTS.json')
    for path in (sites.SITES, sites.HGNC.resolve()):
        if base.digest(path) != reference['external_sources'][str(path)]:
            raise ValueError('Auxiliary source changed')
    if base.digest(sites.REFERENCE / 'dictionary.json') != reference['files']['dictionary.json']:
        raise ValueError('Gene dictionary changed')
    dg_path = sites.DRUG_INPUTS / 'drug_gene_expanded.npz'
    if base.digest(dg_path) != base.read_json(sites.DRUG_INPUTS / 'INPUTS.json')['files'][dg_path.name]:
        raise ValueError('Drug-target evidence changed')
    targets = {dictionary['genes'][i] for i in sparse.load_npz(dg_path).indices}
    genes = {}
    with gzip.open(sites.HGNC, 'rt') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            accessions = sorted(a for a in row['refseq_accession'].split('|') if a.startswith('NM_'))[:2]
            gene = row['ensembl_gene_id'].split('.')[0]
            if row['status'] == 'Approved' and row['locus_group'] == 'protein-coding gene' and gene in targets and accessions:
                genes[row['entrez_id']] = dict(gene_id=gene, symbol=row['symbol'], accessions=accessions)
    groups = {}
    with sites.SITES.open(encoding='utf-8-sig') as handle:
        for line, row in enumerate(csv.DictReader(handle), 2):
            name, entrez = row['miRNA'].upper(), row['Target Gene (Entrez ID)'].removesuffix('.0')
            seq = row['Target Site'].strip().upper().replace('U', 'T')
            if row['Species (miRNA)'] != 'hsa' or row['Species (Target Gene)'] != 'hsa':
                continue
            if name not in rna or entrez not in genes or not 12 <= len(seq) <= 50 or not set(seq) <= set('ACGT'):
                continue
            mature = rna[name]['sequence'].upper().replace('U', 'T')
            if not 18 <= len(mature) <= 28 or not set(mature) <= set('ACGT'):
                continue
            key = (entrez, name, seq)
            item = groups.setdefault(key, dict(rna_name=name, rna_index=rna[name]['index'],
                mirna_sequence=mature, site_sequence=seq, entrez_id=entrez,
                gene_id=genes[entrez]['gene_id'], gene_symbol=genes[entrez]['symbol'],
                pmids=set(), source_lines=[], support_types=set()))
            item['pmids'].update(row['References (PMID)'].replace(',', ';').split(';'))
            item['source_lines'].append(line)
            item['support_types'].add(row['Support Type'])
    requested = sorted({a for key in groups for a in genes[key[0]]['accessions']})
    p = begin(out, dict(selection='All eligible unique RNA/gene/site tuples; no label-dependent selection or per-gene cap.',
        sequence_policy='Up to two NM accessions per gene, unique forward exact site match and real centered 50nt flanks.',
        requested_accessions=requested, candidate_sites=len(groups), no_association_labels=True,
        sources={str(path.resolve()): base.digest(path) for path in
                 (sites.SITES, sites.HGNC, RNA/'MiDrug_sequences.json', dg_path, sites.REFERENCE/'dictionary.json')}))
    transcripts = {}
    cache = out / 'refseq'
    cache.mkdir(exist_ok=True)
    for start in range(0, len(requested), 60):
        ids = requested[start:start+60]
        url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?' + urllib.parse.urlencode(
            dict(db='nuccore', id=','.join(ids), rettype='fasta', retmode='text'))
        path = cache / ('batch_%04d.fasta' % start)
        meta = path.with_suffix('.json')
        if path.exists() and meta.exists():
            m = base.read_json(meta)
            if m['url'] != url or m['sha256'] != base.digest(path):
                raise ValueError('RefSeq cache changed')
            raw = path.read_bytes()
        else:
            for attempt in range(3):
                try:
                    req = urllib.request.Request(url, headers={'User-Agent': 'NcRNAFunctionalExperiment/1.0'})
                    with urllib.request.urlopen(req, timeout=35) as response:
                        raw = response.read()
                    if not raw.startswith(b'>'):
                        raise ValueError('NCBI response is not FASTA')
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(1 + attempt)
            path.write_bytes(raw)
            base.write_json(meta, dict(url=url, sha256=base.digest(path), retrieved_at_utc=datetime.now(timezone.utc).isoformat()))
        batch = sites.fasta_records(raw.decode())
        if any(not requested_accession(name, ids) for name in batch):
            raise ValueError('Unexpected RefSeq accession')
        transcripts.update(batch)
        print('REFSEQ', min(start+60, len(requested)), '/', len(requested), flush=True)
    rows, rejected = [], Counter()
    for key, item in sorted(groups.items()):
        accepted = None
        for acc in genes[key[0]]['accessions']:
            name = resolve_accession(acc, transcripts)
            if name is None:
                rejected['missing_accession_attempts'] += 1
                continue
            window, reason = sites.exact_window(transcripts[name], item['site_sequence'])
            if reason:
                rejected[reason + '_attempts'] += 1
                continue
            accepted = dict(item, refseq_transcript=name, **window)
            break
        if accepted is None:
            rejected['rejected_unique_sites'] += 1
            continue
        for field in ('pmids', 'source_lines', 'support_types'):
            accepted[field] = ';'.join(str(x).strip() for x in sorted(set(accepted[field])) if str(x).strip())
        accepted['site_id'] = base.json_digest(list(key))[:24]
        rows.append(accepted)
    if not rows:
        raise ValueError('No verified target windows')
    with (out/'sites.tsv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    finish(out, p, ['sites.tsv'] + [str(x.relative_to(out)) for x in sorted(cache.iterdir())],
           sites=len(rows), mirnas=len({r['rna_index'] for r in rows}), genes=len({r['gene_id'] for r in rows}),
           requested_transcripts=len(requested), returned_transcripts=len(transcripts), rejected=dict(rejected),
           target_padding_used=False, original_isoform_identity_claimed=False)


def site_rows(out):
    directory = out / 'windows'
    verify(directory)
    with (directory/'sites.tsv').open() as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def moments(local, weights):
    """Weighted means and within-RNA SDs, with explicit all-missing zeros."""
    weights = np.asarray(weights, np.float64)
    denom = weights.sum(1, keepdims=True)
    w = np.divide(weights, denom, out=np.zeros_like(weights), where=denom > 0)
    mean = np.einsum('ns,nsd->nd', w, local, optimize=True)
    second = np.einsum('ns,nsd->nd', w, np.square(local.astype(np.float64)), optimize=True)
    return mean.astype(np.float32), np.sqrt(np.maximum(second - mean**2, 0)).astype(np.float32)


def parnet(args):
    import torch
    from improve.rna_functional_pilot import parnet_loader, state_digest
    from improve.tabicl_reli_inputs import local_sites
    records, generic = verified_sequences('LncDrug'), verified_rnafm('LncDrug')
    eligible = np.array([r['status'] == 'annotated_full_transcript' and len(r['sequence']) >= 200
                         and set(r['sequence'].upper()) <= set('ACGTU') for r in records])
    directory = Path(args.out_dir)/'parnet'
    loader, checkpoint, compat = parnet_loader()
    p = begin(directory, dict(checkpoint_sha256=base.digest(checkpoint),
        source_sha256=base.digest(compat/'SOURCE_FILES.json'), rnafm_sha256=base.digest(RNA/'rna_LncDrug.npz'),
        eligibility='All complete annotated ACGTU lncRNA transcripts >=200nt; no full-transcript length cap.',
        representation='Exact same <=510nt encoder windows, 8nt pooling intervals and weights as cached RNA-FM. Capture PARNET pre-head hidden states. No RNA-FM/PARNET input padding.',
        no_association_labels=True, sequences_sha256=base.digest(RNA/'LncDrug_sequences.json')))
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = loader().eval().to(args.device)
    before = state_digest(model)
    from parnet.utils import sequence_to_onehot
    captured = {}
    def hook(module, inputs):
        captured['hidden'] = inputs[0]
    handle = model.head.register_forward_pre_hook(hook)
    local = np.zeros((len(records), 64, 512), np.float32)
    jobs = defaultdict(list)
    for i in np.flatnonzero(eligible):
        intervals, weights, starts = local_sites(len(records[i]['sequence']))
        np.testing.assert_array_equal(generic['positions'][i, :len(intervals)], intervals)
        np.testing.assert_allclose(generic['weights'][i, :len(weights)], weights)
        for j, ((a, b), start) in enumerate(zip(intervals, starts)):
            jobs[(int(i), start)].append((j, a-start, b-start))
    # Equal-length batches avoid introducing sequence padding into the CNN.
    lengths = defaultdict(list)
    for key in jobs:
        lengths[len(records[key[0]]['sequence'][key[1]:key[1]+510])].append(key)
    def infer(keys):
        x = torch.stack([sequence_to_onehot(records[i]['sequence'][start:start+510].replace('U','T')).float()
                         for i, start in keys]).to(args.device)
        with torch.inference_mode():
            model({'sequence': x})
            hidden = captured.pop('hidden')
            if hidden.shape[-1] != x.shape[-1] or not torch.isfinite(hidden).all():
                raise ValueError('PARNET positions or values changed')
            return hidden.cpu().numpy()
    start_time, completed = time.monotonic(), 0
    for keys in lengths.values():
        for start in range(0, len(keys), args.batch_size):
            batch = keys[start:start+args.batch_size]
            hidden = infer(batch)
            for b, key in enumerate(batch):
                for j, a, end in jobs[key]:
                    local[key[0], j] = hidden[b, :, a:end].mean(-1)
            completed += len(batch)
            if completed % 256 < args.batch_size:
                print('PARNET_MATCHED', completed, '/', len(jobs), flush=True)
    sample = list(jobs)[::max(1, len(jobs)//12)][:12]
    max_error = 0.
    for key in sample[::-1]:
        hidden = infer([key])[0]
        for j, a, b in jobs[key]:
            replay = hidden[:, a:b].mean(-1)
            np.testing.assert_allclose(replay, local[key[0], j], atol=2e-5, rtol=2e-5)
            max_error = max(max_error, float(np.max(np.abs(replay-local[key[0], j]))))
    if state_digest(model) != before:
        raise ValueError('Frozen PARNET state changed')
    handle.remove()
    weights = generic['weights'] * eligible[:, None]
    mean, std = moments(local, weights)
    np.savez_compressed(directory/'representations.npz', local=local, weights=weights,
                        mean=mean, std=std, present=eligible, names=generic['names'])
    finish(directory, p, ['representations.npz'], rnas=int(eligible.sum()), total_rnas=len(records),
           windows=len(jobs), local_units=int((weights>0).sum()), state_sha256=before,
           query_order_and_single_batch_check=True, replay_max_abs=max_error, seconds=time.monotonic()-start_time,
           limitation='Historical public PARNET checkpoint; eCLIP pretraining in K562/HepG2. Not measured regulation or broad cell-state validation.')


def mirbind(args):
    import torch
    from improve.rna_functional_pilot import verify_model_files, module_file, state_digest
    from improve.rna_functional_sources import ROOT, REPOS
    out = Path(args.out_dir)
    rows = site_rows(out)
    source, directory = ROOT/'BioGeMT__miRBind_2.0', out/'mirbind'
    verify_model_files(source, 'SOURCE_FILES.json')
    verify_model_files(source, 'WEIGHT_FILES.json')
    shared = source/'code/pairwise_binding_site_model/shared'
    models = module_file('experiment_mirbind_models', shared/'models.py')
    encoding = module_file('experiment_mirbind_encoding', shared/'encoding.py')
    constants = module_file('experiment_mirbind_constants', shared/'constants.py')
    checkpoint = source/'models/pairwise_onehot_model_20260105_200141.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    params = saved['model_params']
    p = begin(directory, dict(code_commit=REPOS['mirbind'][1], checkpoint_sha256=base.digest(checkpoint),
        windows_sha256=base.digest(out/'windows/sites.tsv'), no_association_labels=True,
        representation='Post-activation 30D hidden representation of each verified 50nt target window and its mature miRNA; miRNA uses official N padding to 28nt, never target padding.', model_params=params))
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = models.PairwiseOneHotCNN(**params)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model = model.eval().to(args.device)
    before, pair_index, captured = state_digest(model), constants.get_pair_to_index(), {}
    def hook(module, inputs, output):
        captured['hidden'] = output
    handle = model.dropout_fc.register_forward_hook(hook)
    def infer(indices):
        encoded = []
        for i in indices:
            r = rows[i]
            if len(r['target_sequence']) != params['target_length'] or len(r['mirna_sequence']) > params['mirna_length']:
                raise ValueError('Invalid real target/mature sequence length')
            mature = encoding.pad_or_trim(r['mirna_sequence'], params['mirna_length'])
            code = encoding.encode_complementarity(r['target_sequence'], mature, params['target_length'],
                        params['mirna_length'], pair_index, len(pair_index))
            encoded.append(torch.nn.functional.one_hot(torch.tensor(code, dtype=torch.long), len(pair_index)+1).float())
        with torch.inference_mode():
            score = model(torch.stack(encoded).to(args.device)).ravel().cpu().numpy()
            return score, captured.pop('hidden').cpu().numpy()
    scores, features = np.empty(len(rows), np.float32), np.empty((len(rows), 30), np.float32)
    for start in range(0, len(rows), args.batch_size):
        idx = np.arange(start, min(start+args.batch_size, len(rows)))
        scores[idx], features[idx] = infer(idx)
    check = np.linspace(0, len(rows)-1, min(32, len(rows)), dtype=int)[::-1]
    replay, hidden = infer(check)
    np.testing.assert_allclose(replay, scores[check], atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(hidden, features[check], atol=2e-5, rtol=2e-5)
    if state_digest(model) != before:
        raise ValueError('Frozen miRBind state changed')
    handle.remove()
    np.savez_compressed(directory/'representations.npz', features=features, binding_scores=scores,
                        rna_indices=np.array([int(r['rna_index']) for r in rows]), site_ids=np.array([r['site_id'] for r in rows]))
    finish(directory, p, ['representations.npz'], sites=len(rows), mirnas=len({r['rna_index'] for r in rows}),
           state_sha256=before, replay_max_abs=float(np.max(np.abs(hidden-features[check]))),
           limitation='Binding-site model output, not the separate transferred repression model or measured suppression strength.')


def generic_sites(args):
    """Run in the existing ncRNAdrug environment; do not modify TabICL deps."""
    import torch
    from multimolecule import RnaFmModel, RnaTokenizer
    from multimolecule.models.rnafm.configuration_rnafm import RnaFmConfig
    from safetensors import safe_open
    out = Path(args.out_dir)
    rows = site_rows(out)
    directory = out/'generic_sites'
    checkpoint = sites.ROOT/'incrd-run/models/rnafm'
    old = base.read_json(RNA/'rna_MiDrug.json')
    if base.digest(checkpoint/'model.safetensors') != old['encoder_sha256']:
        raise ValueError('Generic RNA encoder changed')
    sequences = sorted({r['target_sequence'].replace('T','U') for r in rows} |
                       {r['mirna_sequence'].replace('T','U') for r in rows})
    p = begin(directory, dict(checkpoint_sha256=old['encoder_sha256'], windows_sha256=base.digest(out/'windows/sites.tsv'),
        representation='RNA-FM full-sequence mean for the exact same mature miRNA and real 50nt target windows seen by miRBind. Concatenate mature and target means per site.', no_association_labels=True))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20260914)
    model = RnaFmModel(RnaFmConfig.from_pretrained(str(checkpoint)))
    with safe_open(str(checkpoint/'model.safetensors'), framework='pt') as handle:
        state = {k[len('model.'):]: handle.get_tensor(k) for k in handle.keys() if k.startswith('model.')}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if any(not k.startswith('pooler.') for k in missing) or unexpected:
        raise ValueError('RNA-FM backbone weights did not load')
    del state
    model = model.eval().requires_grad_(False).to(args.device)
    tokenizer = RnaTokenizer.from_pretrained(str(checkpoint))
    table = np.empty((len(sequences), 640), np.float32)
    def infer(indices):
        seqs = [sequences[i] for i in indices]
        encoded = tokenizer(seqs, return_tensors='pt', padding=True, truncation=False)
        for i, seq in enumerate(seqs):
            if int(encoded['attention_mask'][i].sum()) != len(seq)+2:
                raise ValueError('RNA tokenizer changed nucleotide positions')
        with torch.no_grad():
            hidden = model(input_ids=encoded['input_ids'].to(args.device),
                           attention_mask=encoded['attention_mask'].to(args.device)).last_hidden_state
        return np.stack([hidden[i, 1:len(seq)+1].mean(0).float().cpu().numpy() for i, seq in enumerate(seqs)])
    for start in range(0, len(sequences), args.batch_size):
        ids = np.arange(start, min(start+args.batch_size, len(sequences)))
        table[ids] = infer(ids)
        if start % (args.batch_size*20) == 0:
            print('RNAFM_SITES', min(start+args.batch_size,len(sequences)), '/', len(sequences), flush=True)
    check = np.linspace(0, len(sequences)-1, min(24,len(sequences)), dtype=int)[::-1]
    replay = infer(check)
    np.testing.assert_allclose(replay, table[check], atol=2e-5, rtol=2e-5)
    index = {s: i for i, s in enumerate(sequences)}
    source_index = np.array([[index[r['mirna_sequence'].replace('T','U')], index[r['target_sequence'].replace('T','U')]] for r in rows])
    np.savez_compressed(directory/'representations.npz', sequence_features=table, sequences=np.array(sequences),
                        site_sequence_indices=source_index, site_ids=np.array([r['site_id'] for r in rows]))
    finish(directory, p, ['representations.npz'], unique_sequences=len(sequences), sites=len(rows),
           replay_max_abs=float(np.max(np.abs(replay-table[check]))), unused_pooler_keys=list(missing))


def gc_fraction(sequence):
    sequence = sequence.upper().replace('U', 'T')
    return (sequence.count('G')+sequence.count('C')) / max(len(sequence), 1)


def assemble(args):
    out = Path(args.out_dir)
    dependencies = {name: verify(out/name) for name in ('windows','parnet','mirbind','generic_sites')}
    p = begin(out/'assembled', dict(no_association_labels=True,
        dependencies={name: base.digest(out/name/'VALIDATION.json') for name in dependencies},
        simple_columns=['sequence_present','complete_sequence','log_length','gc_fraction','ambiguous_fraction',
                        'functional_present','log_local_unit_count','mean_local_unit_gc'],
        comparison='Generic and functional payloads use identical entity coverage. PCA is deferred to each downstream support context. Uncovered entities are retained with zero payload.',
        functional_mean='Weighted mean of the same local functional units.',
        functional_moments='Concatenation of the weighted mean and within-RNA SD of those exact units.'))
    rows = site_rows(out)
    files, coverages = [], {}
    for task in ('MiDrug', 'LncDrug'):
        records = verified_sequences(task)
        n = len(records)
        if task == 'MiDrug':
            with np.load(out/'mirbind/representations.npz') as z:
                f, site_ids = z['features'], z['site_ids']
                rid = z['rna_indices']
            with np.load(out/'generic_sites/representations.npz') as z:
                np.testing.assert_array_equal(site_ids, z['site_ids'])
                generic = z['sequence_features'][z['site_sequence_indices']].reshape(len(rows), -1)
            gmean, fmean, fstd = np.zeros((n,generic.shape[1]),np.float32), np.zeros((n,30),np.float32), np.zeros((n,30),np.float32)
            count, local_gc = np.zeros(n, np.int32), np.zeros(n,np.float32)
            for i in np.unique(rid):
                selected = np.flatnonzero(rid==i)
                gmean[i], fmean[i], fstd[i] = generic[selected].mean(0), f[selected].mean(0), f[selected].std(0)
                count[i], local_gc[i] = len(selected), np.mean([gc_fraction(rows[j]['target_sequence']) for j in selected])
            present = count > 0
        else:
            rnafm = verified_rnafm(task)
            with np.load(out/'parnet/representations.npz') as z:
                fmean, fstd, present, weights = z['mean'], z['std'], z['present'], z['weights']
                np.testing.assert_array_equal(z['names'], rnafm['names'])
            gmean, _ = moments(rnafm['local'], weights)
            count, local_gc = (weights>0).sum(1), np.zeros(n,np.float32)
            for i in np.flatnonzero(present):
                seq = records[i]['sequence']
                gc = [gc_fraction(seq[a:b]) for a,b in rnafm['positions'][i,:count[i]]]
                local_gc[i] = np.average(gc, weights=weights[i,:count[i]])
        simple = np.array([[bool(r['sequence']), task=='MiDrug' or r['status']=='annotated_full_transcript',
            np.log1p(len(r['sequence'])), gc_fraction(r['sequence']),
            sum(c not in 'ACGTU' for c in r['sequence'].upper())/max(len(r['sequence']),1),
            present[i], np.log1p(count[i]), local_gc[i]] for i,r in enumerate(records)], np.float32)
        name = task+'.npz'
        np.savez_compressed(out/'assembled'/name, simple=simple, generic=gmean,
            functional_mean=fmean, functional_moments=np.concatenate([fmean,fstd],axis=1),
            present=present, unit_counts=count, names=np.array([r['name'] for r in records]))
        for a in (simple, gmean, fmean, fstd):
            if not np.isfinite(a).all():
                raise ValueError('Nonfinite assembled features')
        files.append(name)
        coverages[task] = dict(rnas=n, covered=int(present.sum()), unit_count=int(count.sum()),
            generic_dimension=gmean.shape[1], functional_mean_dimension=fmean.shape[1],
            functional_moments_dimension=fmean.shape[1]*2)
    finish(out/'assembled', p, files, coverage=coverages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['windows','parnet','mirbind','generic_sites','assemble'])
    parser.add_argument('--out-dir', default=str(DEFAULT_OUT))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
