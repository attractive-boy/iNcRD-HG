"""Download and assemble label-free RNA/gene/pathway evidence for both tasks.

ENCORI observations are binding evidence, not validated regulatory directions.
All source responses must finish before they enter the feature build.
"""
import argparse
import concurrent.futures
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import requests
from scipy import sparse


DEFAULT_OUT = 'improve/results/tabicl_pathway_inputs_20260913'
NET = Path('/data1/licheng/incrd-hg/incrd-run/2.construct_network')
CHEMBL = NET.parent / '14.drug/chembl.json'
IDENTITY = Path('improve/research/rna_gene_mapping_20260913/rna_identity.tsv')
REQUEST_LOCK = threading.Lock()
NEXT_REQUEST_AT = 0.0
URLS = {
    'encori_mi': 'https://rnasysu.com/encori/api/miRNATarget/?assembly=hg38&geneType=mRNA&miRNA=all&clipExpNum=1&degraExpNum=0&pancancerNum=0&programNum=0&target=all&cellType=all',
    'encori_lnc': 'https://rnasysu.com/encori/api/RNARNA/?assembly=hg38&geneType=lncRNA&RNA=all&interNum=1&expNum=1&cellType=all',
    'reactome_gene': 'https://reactome.org/download/current/Ensembl2Reactome.txt',
    'hgnc': 'https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def fetch(dest, name, url):
    dest.mkdir(parents=True, exist_ok=True)
    path, manifest = dest / (name + '.txt.gz'), dest / (name + '.json')
    if manifest.exists():
        m = json.loads(manifest.read_text())
        if m.get('complete') and m['url'] == url and digest(path) == m['file_sha256']:
            print('CACHED', name, m['bytes'], flush=True)
            return m
    errors = []
    for attempt in range(2):
        start = time.monotonic()
        h, size, next_log = hashlib.sha256(), 0, 32 * 2**20
        partial = path.with_suffix('.partial.gz')
        try:
            global NEXT_REQUEST_AT
            with REQUEST_LOCK:
                delay = max(0., NEXT_REQUEST_AT - time.monotonic())
                if delay:
                    time.sleep(delay)
                NEXT_REQUEST_AT = time.monotonic() + 0.6
            with requests.get(url, stream=True, timeout=(20, 90), headers={
                    'User-Agent': 'iNcRD-HG-public-annotation-audit/1.0'}) as r:
                r.raise_for_status()
                with gzip.open(partial, 'wb', compresslevel=3) as f:
                    for block in r.iter_content(2**18):
                        if not block:
                            continue
                        if size == 0 and b'<html' in block[:500].lower():
                            raise ValueError('HTML response is not a data table')
                        h.update(block)
                        f.write(block)
                        size += len(block)
                        if size >= next_log:
                            print('DOWNLOADING', name, round(size/2**20, 1), 'MiB', flush=True)
                            next_log += 32 * 2**20
                if size == 0:
                    raise ValueError('Empty data response')
                partial.replace(path)
                m = dict(name=name, url=url, final_url=r.url, status=r.status_code,
                         downloaded_at_utc=utc(), complete=True, bytes=size,
                         response_sha256=h.hexdigest(), file_sha256=digest(path),
                         content_type=r.headers.get('Content-Type'),
                         etag=r.headers.get('ETag'), last_modified=r.headers.get('Last-Modified'),
                         elapsed_seconds=time.monotonic()-start, previous_errors=errors)
                write_json(manifest, m)
                print('DOWNLOADED', name, size, 'bytes', flush=True)
                return m
        except Exception as exc:
            errors.append(dict(attempt=attempt + 1, error=str(exc), received_bytes=size, at_utc=utc()))
            print('DOWNLOAD_ERROR', name, errors[-1], flush=True)
            time.sleep(2)
    write_json(manifest, dict(name=name, url=url, complete=False, errors=errors))
    raise RuntimeError('Could not finish public source: ' + name)


def download(out):
    out = Path(out)
    if (out/'INPUTS.json').exists():
        raise ValueError('Frozen inputs exist; use a new directory to refresh annotations')
    source = out / 'sources'
    source.mkdir(parents=True, exist_ok=True)
    version_url = 'https://reactome.org/ContentService/data/database/version'
    version = requests.get(version_url, timeout=30)
    version.raise_for_status()
    write_json(source/'reactome_version.json', dict(version=version.text.strip(), url=version_url, at_utc=utc()))
    # The ENCORI all-RNA endpoints have returned a PHP execution timeout or a
    # database failure inside HTTP 200. Query the project entities separately.
    jobs = {k:v for k,v in URLS.items() if not k.startswith('encori_')}
    drug = json.loads(CHEMBL.read_text())
    accessions = sorted({a for target in drug['target'].values() for a in target['accessions']})
    for i in range(0, len(accessions), 100):
        params = dict(query='(' + ' OR '.join('accession:' + a for a in accessions[i:i+100]) + ')',
                      format='json', fields='accession,xref_ensembl,organism_id,gene_primary', size=500)
        url = requests.Request('GET', 'https://rest.uniprot.org/uniprotkb/search', params=params).prepare().url
        jobs['uniprot_' + str(i//100)] = url
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch, source, name, url): name for name, url in jobs.items()}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures.append(dict(source=futures[future], error=str(exc)))
    write_json(out/'DOWNLOAD_STATUS.json', dict(at_utc=utc(), failures=failures, sources=list(jobs)))
    if failures:
        raise RuntimeError(str(failures))
    download_entities(out)


def inspect_encori(path, kind):
    """Reject server-side partial responses, including errors inside HTTP 200."""
    identifiers, rows, unavailable, header = set(), 0, False, None
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            fields = line.rstrip('\r\n').split('\t')
            if header is None:
                header = fields
                expected = 'miRNAid' if kind == 'encori_mi' else 'geneID'
                if header[0] != expected:
                    raise ValueError('Unexpected ENCORI header')
                continue
            if line.startswith('The "') and 'parameter' in line and 'not available' in line:
                unavailable = True
                continue
            if len(fields) != len(header):
                raise ValueError('Malformed/error line: ' + line[:180])
            if not fields[0].startswith(('ENSG', 'MIMAT')):
                raise ValueError('Invalid source RNA identifier')
            identifiers.add(fields[0].split('.')[0])
            rows += 1
    if header is None or (rows and unavailable):
        raise ValueError('Missing or mixed ENCORI response')
    return dict(rows=rows, identifiers=sorted(identifiers),
                query_status='rows' if rows else ('unavailable_name_or_no_annotation' if unavailable else 'empty'))


def download_entities(out):
    out = Path(out)
    with IDENTITY.open() as f:
        identities = list(csv.DictReader(f, delimiter='\t'))
    hgnc = list(source_rows(out, 'hgnc'))
    gene_names = {r['ensembl_gene_id']:r['symbol'] for r in hgnc if r.get('ensembl_gene_id')}
    seqs = json.loads(Path('improve/results/tabicl_reli_inputs_20260912/MiDrug_sequences.json').read_text())
    mirna_names = {r['name']:r['source_name'] for r in seqs}
    source = out/'sources'
    support_file = out/'SOURCE_SUPPORT_POLICY.json'
    support_policy = json.loads(support_file.read_text()) if support_file.exists() else {}
    overrides = {(r['task'],r['rna_name']):r for r in support_policy.get('query_overrides',[])}
    jobs = []
    for r in identities:
        kind = 'encori_mi' if r['task'] == 'MiDrug' else 'encori_lnc'
        primary = mirna_names[r['rna_name']] if r['task'] == 'MiDrug' else r['rna_name']
        names = [primary]
        if r['task'] == 'MiDrug':
            historical_name = r['rna_name'].lower().replace('hsa-mir-', 'hsa-miR-')
            if historical_name != primary:
                names.append(historical_name)
        else:
            canonical = gene_names.get(r['rna_stable_id'])
            if canonical and canonical != primary:
                names.append(canonical)
        jobs.append((r, kind, names))

    def query(job):
        row, kind, names = job
        override = overrides.get((row['task'],row['rna_name']))
        if override:
            path = out/override['path']
            meta = json.loads(path.with_suffix('').with_suffix('.json').read_text())
            check = inspect_encori(path,kind)
            if not meta['complete'] or digest(path) != meta['file_sha256']:
                raise ValueError('Unverified alternate complete query')
            if check['identifiers'] != [row['rna_stable_id']]:
                raise ValueError('Alternate query RNA identity mismatch')
            return [dict(task=row['task'],rna_name=row['rna_name'],
                         rna_node_index_0based=int(row['rna_node_index_0based']),
                         expected_id=row['rna_stable_id'],query_name=row['rna_name'],
                         relative_path=override['path'],retrieval_override=True,**check)]
        stem = str(row['rna_node_index_0based']).zfill(4)
        records = []
        for alias_idx, name in enumerate(names):
            dest = source/'entity_queries'/kind
            token = stem + '_' + str(alias_idx)
            endpoint = 'miRNATarget' if kind == 'encori_mi' else 'RNARNA'
            params = (dict(assembly='hg38', geneType='mRNA', miRNA=name, clipExpNum=1,
                           degraExpNum=0, pancancerNum=0, programNum=0, target='all', cellType='all')
                      if kind == 'encori_mi' else dict(assembly='hg38', geneType='lncRNA',
                                                     RNA=name, interNum=1, expNum=1, cellType='all'))
            url = requests.Request('GET', 'https://rnasysu.com/encori/api/'+endpoint+'/', params=params).prepare().url
            last_error = None
            for attempt in range(2):
                try:
                    meta = fetch(dest, token, url)
                    check = inspect_encori(dest/(token+'.txt.gz'), kind)
                    meta.update(check)
                    write_json(dest/(token+'.json'), meta)
                    record = dict(task=row['task'], rna_name=row['rna_name'],
                                  rna_node_index_0based=int(row['rna_node_index_0based']),
                                  expected_id=row['rna_stable_id'], query_name=name,
                                  relative_path=str((dest/(token+'.txt.gz')).relative_to(out)), **check)
                    if row['rna_stable_id'] and check['rows'] and check['identifiers'] != [row['rna_stable_id']]:
                        record['query_status'] = 'identity_mismatch'
                    records.append(record)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = str(exc)
                    mpath = dest/(token+'.json')
                    if mpath.exists():
                        meta = json.loads(mpath.read_text())
                        meta.update(complete=False, data_validation_error=last_error)
                        write_json(mpath, meta)
                    time.sleep(2)
            if last_error:
                raise RuntimeError(row['rna_name'] + ': ' + last_error)
            if records[-1]['query_status'] == 'rows':
                break
        return records

    results, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(query, job):job[0]['rna_name'] for job in jobs}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                results.extend(future.result())
            except Exception as exc:
                failures.append(dict(rna=futures[future], error=str(exc)))
            if i % 50 == 0:
                print('ENTITY_PROGRESS', i, '/', len(jobs), 'failures', len(failures), flush=True)
    results.sort(key=lambda r:(r['task'],r['rna_node_index_0based'],r['query_name']))
    previous = out/'ENTITY_QUERIES.json'
    if previous.exists():
        archive = out/'query_passes'
        archive.mkdir(exist_ok=True)
        previous.replace(archive/('queries_'+str(time.time_ns())+'.json'))
    write_json(out/'ENTITY_QUERIES.json', dict(created_at_utc=utc(), entities=len(jobs),
               results=results, failures=failures))
    if failures:
        raise RuntimeError(f'{len(failures)} unfinished RNA queries; see ENTITY_QUERIES.json')
    for kind, task in [('encori_mi','MiDrug'),('encori_lnc','LncDrug')]:
        path, meta = source/(kind+'.txt.gz'), source/(kind+'.json')
        if path.exists():
            old = json.loads(meta.read_text())
            if old.get('assembly_method') != 'validated_per_entity_queries':
                failed = source/'failed_bulk'
                failed.mkdir(exist_ok=True)
                old.update(complete=False, reason='HTTP body contained database/PHP error; not usable as a complete data source.')
                write_json(meta, old)
                path.replace(failed/path.name)
                meta.replace(failed/meta.name)
        header_written, total = False, 0
        with gzip.open(path, 'wt', compresslevel=3) as f:
            for record in results:
                if record['task'] != task or record['query_status'] != 'rows':
                    continue
                with gzip.open(out/record['relative_path'],'rt') as raw:
                    seen_header = False
                    for line in raw:
                        if line.startswith('#') or not line.strip():
                            continue
                        if not seen_header:
                            if not header_written:
                                f.write(line)
                                header_written = True
                            seen_header = True
                            continue
                        f.write(line)
                        total += 1
        if not header_written:
            raise ValueError('No valid entity data for ' + kind)
        check = inspect_encori(path,kind)
        if check['rows'] != total:
            raise ValueError('Merged row count mismatch')
        write_json(meta, dict(complete=True, assembly_method='validated_per_entity_queries',
                             url=URLS[kind], file_sha256=digest(path), rows=total,
                             query_manifest_sha256=digest(out/'ENTITY_QUERIES.json'), at_utc=utc()))
        print('MERGED_VALIDATED_SOURCE', kind, total, flush=True)


def source_rows(out, name, header=True):
    meta = json.loads((out/'sources'/f'{name}.json').read_text())
    path = out/'sources'/f'{name}.txt.gz'
    if not meta['complete'] or digest(path) != meta['file_sha256']:
        raise ValueError('Incomplete or changed source: ' + name)
    with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as f:
        rows = (line for line in f if not line.startswith('#'))
        reader = csv.DictReader(rows, delimiter='\t', quoting=csv.QUOTE_NONE) if header else csv.reader(rows, delimiter='\t', quoting=csv.QUOTE_NONE)
        yield from reader


def write_tsv(path, rows, fields):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def normalized(matrix):
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(1)).ravel())
    return sparse.diags(np.divide(1., norms, out=np.zeros_like(norms), where=norms > 0)).dot(matrix).tocsr()


def row_prob(matrix):
    sums = np.asarray(matrix.sum(1)).ravel()
    return sparse.diags(np.divide(1., sums, out=np.zeros_like(sums), where=sums > 0)).dot(matrix).tocsr()


def build(out):
    out = Path(out)
    if (out/'INPUTS.json').exists():
        raise ValueError('Inputs already frozen; use a new output directory for changes')
    support_file = out/'SOURCE_SUPPORT_POLICY.json'
    support_policy = (json.loads(support_file.read_text()) if support_file.exists() else
                      dict(min_mirna_clip_experiments=1,min_lnc_interactions=1,min_lnc_experiments=1))
    with IDENTITY.open() as f:
        identities = list(csv.DictReader(f, delimiter='\t'))
    hgnc = list(source_rows(out, 'hgnc'))
    lookup = {tier: {} for tier in ('symbol', 'prev_symbol', 'alias_symbol')}
    for row in hgnc:
        if row.get('status') != 'Approved' or not row.get('ensembl_gene_id', '').startswith('ENSG'):
            continue
        for tier in lookup:
            for name in row.get(tier, '').split('|'):
                if name:
                    lookup[tier].setdefault(name.upper(), {})[row['ensembl_gene_id']] = row
    resolutions = []
    for r in identities:
        if r['task'] != 'LncDrug' or r['rna_stable_id']:
            continue
        for tier in lookup:
            matches = lookup[tier].get(r['rna_name'].upper(), {})
            if not matches:
                continue
            detail = dict(rna_name=r['rna_name'], tier=tier, matches=sorted(matches))
            if len(matches) == 1:
                gene, annot = next(iter(matches.items()))
                r.update(rna_stable_id=gene, rna_stable_id_raw=gene,
                         identity_status='hgnc_' + tier, source_name=annot['symbol'],
                         source_file='sources/hgnc.txt.gz')
                detail.update(symbol=annot['symbol'], locus_group=annot.get('locus_group'), locus_type=annot.get('locus_type'))
            resolutions.append(detail)
            break
    write_tsv(out/'rna_identity.tsv', identities, list(identities[0]))
    by_task = {t: [r for r in identities if r['task'] == t] for t in ('MiDrug', 'LncDrug')}
    ids = {t: {} for t in by_task}
    for t, rows in by_task.items():
        for r in rows:
            if r['rna_stable_id']:
                ids[t].setdefault(r['rna_stable_id'], []).append(int(r['rna_node_index_0based']))
    edges, parse_audit = {}, {}
    for name, task in [('encori_mi', 'MiDrug'), ('encori_lnc', 'LncDrug')]:
        parsed, matched, filtered = 0, 0, 0
        for record in source_rows(out, name):
            parsed += 1
            if task == 'MiDrug':
                rid, gene = record['miRNAid'], record['geneID']
                accepted = (record['geneType'] == 'protein_coding' and
                            int(record['clipExpNum']) >= support_policy['min_mirna_clip_experiments'])
                target_symbol = record['geneName']
                support = int(record['clipExpNum'])
                context = record.get('cellline/tissue') or ''
            else:
                rid, gene = record['geneID'], record['pairGeneID']
                accepted = (record['geneType'] == 'lncRNA' and record['pairGeneType'] == 'protein_coding'
                            and int(record['interactionNum']) >= support_policy['min_lnc_interactions']
                            and int(record['totalExpNum']) >= support_policy['min_lnc_experiments'])
                target_symbol = record['pairGeneName']
                support = int(record['totalExpNum'])
                context = record.get('CellLine/Tissue') or ''
            if rid not in ids[task]:
                continue
            matched += 1
            if not accepted or not gene.startswith('ENSG'):
                continue
            filtered += 1
            for idx in ids[task][rid]:
                key = task, idx, gene.split('.')[0]
                if key not in edges:
                    edges[key] = dict(task=task, rna_node_index_0based=idx,
                                     rna_name=by_task[task][idx]['rna_name'], rna_stable_id=rid,
                                     target_gene_id=key[2], target_gene_symbol=target_symbol,
                                     relation_type='miRNA_mRNA_binding' if task == 'MiDrug' else 'lncRNA_mRNA_interaction',
                                     evidence_type='CLIP_supported_candidate' if task == 'MiDrug' else 'RNA_RNA_experiment',
                                     effect_direction='unknown', PMID='', source=name,
                                     source_version='hg38; retrieved ' + utc()[:10],
                                     source_rows=0, max_experiments=0, contexts=set())
                e = edges[key]
                e['source_rows'] += 1
                e['max_experiments'] = max(e['max_experiments'], support)
                e['contexts'].update(x for x in context.split(',') if x)
            if parsed % 500000 == 0:
                print('PARSED', name, parsed, flush=True)
        parse_audit[name] = dict(parsed_rows=parsed, identity_matched_rows=matched, accepted_rows=filtered)
        print('PARSED_COMPLETE', name, parse_audit[name], flush=True)
    records = []
    for key in sorted(edges):
        row = dict(edges[key])
        row['context'] = '|'.join(sorted(row.pop('contexts')))
        records.append(row)
    if not records:
        raise ValueError('No RNA–gene observations')
    write_tsv(out/'rna_to_gene.tsv', records, list(records[0]))

    pathway_members, pathway_names = {}, {}
    for row in source_rows(out, 'reactome_gene', header=False):
        if len(row) != 6:
            raise ValueError('Unexpected Reactome annotation row')
        gene, pathway, _, title, evidence, species = row
        if species == 'Homo sapiens' and gene.startswith('ENSG') and pathway.startswith('R-HSA-'):
            pathway_members.setdefault(pathway, set()).add(gene.split('.')[0])
            pathway_names[pathway] = title
    pathways = sorted(p for p, genes in pathway_members.items() if 5 <= len(genes) <= 500)
    if not pathways:
        raise ValueError('No usable human pathways')
    version = json.loads((out/'sources/reactome_version.json').read_text())['version']
    annotation = [dict(gene_id=g, pathway_id=p, pathway_name=pathway_names[p], species='Homo sapiens',
                       source_version=version) for p in pathways for g in sorted(pathway_members[p])]
    write_tsv(out/'gene_to_pathway.tsv', annotation, list(annotation[0]))

    uniprot, uniprot_rows = {}, []
    for meta in sorted((out/'sources').glob('uniprot_*.json')):
        m = json.loads(meta.read_text())
        if not m['complete'] or digest(meta.with_suffix('.txt.gz')) != m['file_sha256']:
            raise ValueError('Incomplete UniProt source')
        with gzip.open(meta.with_suffix('.txt.gz'), 'rt') as f:
            data = json.load(f)
        for entry in data['results']:
            if entry.get('organism', {}).get('taxonId') != 9606:
                continue
            acc = entry['primaryAccession']
            genes = set()
            for ref in entry.get('uniProtKBCrossReferences', []):
                if ref['database'] == 'Ensembl':
                    for prop in ref.get('properties', []):
                        if prop['key'] == 'GeneId' and prop['value'].startswith('ENSG'):
                            genes.add(prop['value'].split('.')[0])
            uniprot[acc] = genes
            uniprot_rows.extend(dict(uniprot_accession=acc, gene_id=g) for g in sorted(genes))
    if not uniprot_rows:
        raise ValueError('No human drug protein-to-gene mappings')
    write_tsv(out/'uniprot_to_gene.tsv', uniprot_rows, list(uniprot_rows[0]))
    drug = json.loads(CHEMBL.read_text())
    with (NET/'Drug.csv').open() as f:
        drug_names = [r['Drug_Name'] for r in csv.DictReader(f)]
    drug_rows, excluded_search = [], []
    for idx, name in enumerate(drug_names):
        molecule = drug['molecule'].get(name)
        if not molecule:
            continue
        if molecule['tier'] not in ('inchikey', 'pref_name'):
            excluded_search.append(name)
            continue
        for mech in drug['mechanism'].get(molecule['chembl_id'], []):
            target = drug['target'].get(mech['target'], {})
            for acc in target.get('accessions', []):
                for gene in sorted(uniprot.get(acc, set())):
                    drug_rows.append(dict(drug_node_index_0based=idx, drug_name=name, gene_id=gene,
                                          chembl_molecule=molecule['chembl_id'], matching_tier=molecule['tier'],
                                          chembl_target=mech['target'], target_type=target['type'],
                                          uniprot=acc, action=mech.get('action'), mechanism=mech.get('moa')))
    if not drug_rows:
        raise ValueError('No exact-identity drug gene evidence')
    write_tsv(out/'drug_to_gene.tsv', drug_rows, list(drug_rows[0]))
    genes = sorted({e['target_gene_id'] for e in records} | {r['gene_id'] for r in drug_rows} |
                   {g for p in pathways for g in pathway_members[p]})
    gene_idx, path_idx = {g:i for i,g in enumerate(genes)}, {p:i for i,p in enumerate(pathways)}
    pr, pc = zip(*[(gene_idx[r['gene_id']], path_idx[r['pathway_id']]) for r in annotation])
    gp = sparse.csr_matrix((np.ones(len(pr), dtype=np.float32), (pr, pc)), shape=(len(genes), len(pathways)))
    dg_keys = sorted({(r['drug_node_index_0based'], gene_idx[r['gene_id']]) for r in drug_rows})
    dr, dc = zip(*dg_keys)
    dg = sparse.csr_matrix((np.ones(len(dr), dtype=np.float32), (dr,dc)), shape=(len(drug_names),len(genes)))
    rgs = {}
    for task in by_task:
        keys = [(e['rna_node_index_0based'], gene_idx[e['target_gene_id']]) for e in records if e['task'] == task]
        rr, rc = zip(*keys)
        rgs[task] = sparse.csr_matrix((np.ones(len(rr),dtype=np.float32),(rr,rc)), shape=(len(by_task[task]),len(genes)))
    lnames = {r['rna_name']: int(r['rna_node_index_0based']) for r in by_task['LncDrug']}
    mnames = {r['rna_name']: int(r['rna_node_index_0based']) for r in by_task['MiDrug']}
    with (NET/'LncMiRNA_interaction.csv').open() as f:
        lm_keys = sorted({(lnames[r['lncRNA']], mnames[r['miRNA']]) for r in csv.DictReader(f)
                          if r['lncRNA'] in lnames and r['miRNA'] in mnames})
    lr, lc = zip(*lm_keys)
    lm = sparse.csr_matrix((np.ones(len(lr),dtype=np.float32),(lr,lc)), shape=(len(lnames),len(mnames)))
    indirect_lnc = row_prob(lm).dot(row_prob(rgs['MiDrug'])).tocsr()
    indirect_mi = row_prob(lm.T.tocsr()).dot(row_prob(rgs['LncDrug'])).tocsr()
    # Apply the same neighbor-program propagation in both directions. These are
    # cross-RNA neighborhood associations, not newly validated direct targets.
    matrices = dict(gene_pathway=gp, drug_gene=dg, LncDrug_indirect_gene=indirect_lnc,
                    MiDrug_indirect_gene=indirect_mi,
                    **{task+'_gene':mat for task,mat in rgs.items()})
    files = []
    for name, mat in matrices.items():
        sparse.save_npz(out/(name+'.npz'), mat)
        files.append(name+'.npz')
    write_json(out/'dictionary.json', dict(genes=genes, pathways=pathways,
               pathway_names=[pathway_names[p] for p in pathways], drug_names=drug_names,
               rna_names={t:[r['rna_name'] for r in by_task[t]] for t in by_task}))
    coverage = {}
    for task, rg in rgs.items():
        counts = np.diff(rg.indptr)
        indirect_mat = matrices[task+'_indirect_gene']
        direct_path = rg.dot(gp)
        union_present = (counts>0) | (np.diff(indirect_mat.indptr)>0)
        coverage[task] = dict(total=rg.shape[0], identity=sum(bool(r['rna_stable_id']) for r in by_task[task]),
                             direct_genes=int((counts>0).sum()), direct_gene_edges=int(rg.nnz),
                             direct_pathways=int((np.diff(direct_path.indptr)>0).sum()),
                             direct_or_indirect_genes=int(union_present.sum()),
                             direct_or_indirect_pathways=int((np.diff((rg+indirect_mat).dot(gp).indptr)>0).sum()),
                             target_count_quantiles=[float(x) for x in np.quantile(counts,[0,.25,.5,.75,1])])
    coverage['Drug'] = dict(total=dg.shape[0], genes=int((np.diff(dg.indptr)>0).sum()),
                            pathways=int((np.diff(dg.dot(gp).indptr)>0).sum()), gene_edges=dg.nnz,
                            excluded_unverified_search_matches=excluded_search)
    audit = dict(created_at_utc=utc(), coverage=coverage, parse_audit=parse_audit, support_policy=support_policy,
                 hgnc_resolutions=resolutions, pathway_count=len(pathways), gene_count=len(genes),
                 pathway_filter='Human Ensembl gene identifiers; direct Reactome annotation; 5–500 genes per pathway.',
                 reactome_version=version, lnc_mirna_edges=len(lm_keys),
                 unavailable_sources=['miRTarBase: TLS handshake failure in prior probe',
                                      'LncTarD: HTTP 403 / certificate hostname mismatch in prior probe'],
                 evidence='ENCORI binding/interaction observations only; indirect candidates separately encoded. No functional regulation claim.',
                 labels_read=False, RNA_drug_edges_read=False, external_evidence_date_policy='Current downloaded annotations, not a temporal holdout.')
    write_json(out/'COVERAGE.json', audit)
    files += ['rna_identity.tsv','rna_to_gene.tsv','gene_to_pathway.tsv','uniprot_to_gene.tsv',
              'drug_to_gene.tsv','dictionary.json','COVERAGE.json']
    if support_file.exists():
        files.append('SOURCE_SUPPORT_POLICY.json')
    source_files = {str(p.relative_to(out)):digest(p) for p in sorted((out/'sources').rglob('*.json'))}
    source_files['ENTITY_QUERIES.json'] = digest(out/'ENTITY_QUERIES.json')
    write_json(out/'INPUTS.json', dict(created_at_utc=utc(), files={f:digest(out/f) for f in files},
                sources=source_files, source_code_sha256=digest(__file__),
                local_sources={str(p):digest(p) for p in [IDENTITY,CHEMBL,NET/'Drug.csv',NET/'LncMiRNA_interaction.csv']}))
    print('INPUTS_FROZEN', json.dumps(coverage,ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('download','build','all'))
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.command in ('download','all'):
        download(args.out_dir)
    if args.command in ('build','all'):
        build(args.out_dir)


if __name__ == '__main__':
    main()
