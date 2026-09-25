"""Freeze curated RNA targets and matched DepMap expression without drug labels.

The uploaded original files are read in place and hashed. Previously frozen
ENCORI/Reactome/HGNC/drug annotations are reused without changing them.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path
import re
import shutil
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata

from improve import tabular_foundation as base
from improve.tabicl_pathway_modules import binary, divide, l2_normalize


ROOT = Path('/data1/licheng/incrd-hg')
UPLOADS = ROOT/'data_external/pathway_20260913'
REFERENCE = Path('improve/results/tabicl_pathway_inputs_20260913')
DEFAULT_OUT = Path('improve/results/tabicl_pathway_context_inputs_20260913')
NET = ROOT/'incrd-run/2.construct_network'
TIERS = ('functional', 'binding', 'association')
WEIGHTS = dict(functional=1., binding=.25, association=.1)
CONTEXT_NAMES = (
    'expression_rna_present', 'expression_drug_target_fraction',
    'rna_drug_target_rank_correlation', 'rna_drug_target_absolute_rank_correlation',
    'rna_drug_target_rank_correlation_rms',
    'within_lineage_target_correlation', 'within_lineage_absolute_target_correlation',
    'within_lineage_target_correlation_rms',
    'drug_program_expression_correlation', 'drug_program_absolute_expression_correlation',
    'within_lineage_drug_program_correlation', 'within_lineage_absolute_drug_program_correlation',
    'shared_evidence_program_expression_agreement', 'target_correlation_lineage_difference',
)
MISSING = {'', 'NA', 'N/A', 'nan', 'None'}


def numeric_id(value):
    value = str(value).strip()
    return value[:-2] if re.fullmatch(r'\d+\.0', value) else value


def stable_id(value):
    value = str(value).strip()
    return value.split('.')[0] if re.fullmatch(r'ENSG\d+(?:\.\d+)?', value) else ''


class GeneLookup:
    """Stable ID and Entrez take precedence over names; conflicting IDs fail."""
    def __init__(self, rows):
        self.ensembl, self.entrez = {}, defaultdict(set)
        self.symbols = {tier: defaultdict(set) for tier in ('symbol', 'prev_symbol', 'alias_symbol')}
        for row in rows:
            gene = stable_id(row.get('ensembl_gene_id', ''))
            if row.get('status') != 'Approved' or not gene:
                continue
            self.ensembl[gene] = row
            if row.get('entrez_id'):
                self.entrez[numeric_id(row['entrez_id'])].add(gene)
            for tier in self.symbols:
                for name in row.get(tier, '').split('|'):
                    if name:
                        self.symbols[tier][name.upper()].add(gene)

    def resolve(self, ensembl='', entrez='', symbol='', protein=False):
        direct = stable_id(ensembl)
        known = {direct} if direct in self.ensembl else set()
        by_entrez = self.entrez.get(numeric_id(entrez), set())
        if len(by_entrez) > 1 or (known and by_entrez and known != by_entrez):
            return '', 'conflicting_or_ambiguous_stable_identifiers'
        candidates = known or by_entrez
        method = 'ensembl' if known else 'entrez'
        if not candidates:
            for tier in self.symbols:
                candidates = self.symbols[tier].get(str(symbol).upper(), set())
                if candidates:
                    method = tier
                    break
        if len(candidates) != 1:
            return '', 'unmapped' if not candidates else 'ambiguous_name'
        gene = next(iter(candidates))
        if protein and self.ensembl[gene]['locus_group'] != 'protein-coding gene':
            return '', 'not_protein_coding'
        return gene, method


def lnctard_tier(row):
    # E/F suffixes describe expression/function directions, not confidence.
    mechanism = row['SearchregulatoryMechanism']
    if mechanism == 'expression association':
        return 'association'
    if mechanism in {'ceRNA or sponge', 'epigenetic regulation', 'interact with protein',
                     'transcriptional regulation', 'interact with mRNA', 'chromatin looping'}:
        return 'functional'
    raise ValueError('Unknown LncTarD mechanism: '+mechanism)


def mti_tier(species_mirna, species_target, support, experiments):
    if species_mirna != 'hsa' or species_target != 'hsa':
        return '', 'non_human_pair'
    if support.startswith('Non-Functional'):
        return '', 'non_functional'
    if support == 'Functional MTI':
        return 'functional', 'accepted'
    if support != 'Functional MTI (Weak)':
        raise ValueError('Unknown MTI support category: '+support)
    # CLIP/CLASH support binding, not a verified regulatory direction.
    if any(token in experiments.upper() for token in ('CLIP', 'CLASH')):
        return 'binding', 'accepted'
    return 'association', 'accepted'


def read_lnctard(path):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Corrupt LncTarD archive')
        raw = archive.read('lnctard2.0.txt')
    # The official file contains 139 invalid UTF-8 bytes in descriptive/name
    # fields. Preserve them losslessly; resolve genes from stable IDs when possible.
    text = raw.decode('utf-8-sig', errors='surrogateescape')
    rows = list(csv.DictReader(io.StringIO(text), delimiter='\t'))
    if any(None in row or None in row.values() for row in rows):
        raise ValueError('Malformed LncTarD table')
    broken = Counter(key for row in rows for key, value in row.items()
                     if any(0xDC80 <= ord(c) <= 0xDCFF for c in value))
    return rows, dict(broken)


def rank_units(values, groups=None):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Expected a finite cells-by-features matrix')
    result = rankdata(values, axis=0, method='average')
    result -= result.mean(0)
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(result):
            raise ValueError('Lineage/cell mismatch')
        for group in np.unique(groups):
            keep = groups == group
            result[keep] -= result[keep].mean(0)
    return unit_columns(result)


def unit_columns(values):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=0)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-12)


def program_projection(gene_pathway):
    gp = binary(gene_pathway)
    degrees = np.asarray(gp.sum(1)).ravel()
    sizes = np.asarray(gp.sum(0)).ravel()
    return (sparse.diags(divide(1., np.sqrt(degrees))) @ gp @
            sparse.diags(divide(1., np.sqrt(sizes)))).tocsr()


def contextual_features(rna, genes, groups, drug_gene, projection, rna_evidence):
    """Symmetric rank correlations; no target-effect direction is imputed."""
    drug_gene = binary(drug_gene)
    nr, nd = rna.shape[1], drug_gene.shape[0]
    gx, rx = rank_units(genes), rank_units(rna)
    gr, rr = rank_units(genes, groups), rank_units(rna, groups)
    valid_genes = np.linalg.norm(gx, axis=0) > 0
    valid_rna = np.linalg.norm(rx, axis=0) > 0
    valid_drug = drug_gene.multiply(valid_genes[None, :]).tocsr()
    drug_counts = np.asarray(drug_gene.sum(1)).ravel()
    measured_counts = np.asarray(valid_drug.sum(1)).ravel()
    target_average = (sparse.diags(divide(1., measured_counts)) @ valid_drug).tocsr()
    global_gene, residual_gene = rx.T @ gx, rr.T @ gr

    def mean_for_drug(cor):
        return np.asarray(target_average.dot(cor.T).T)

    direct = mean_for_drug(global_gene)
    residual = mean_for_drug(residual_gene)
    global_program = rx.T @ unit_columns(gx @ projection)
    residual_program = rr.T @ unit_columns(gr @ projection)
    drug_program = l2_normalize(drug_gene @ projection)
    program_average = sparse.diags(divide(1., np.asarray(drug_program.sum(1)).ravel())) @ drug_program

    def mean_for_program(cor):
        return np.asarray(program_average.dot(cor.T).T)

    prior = l2_normalize(rna_evidence @ projection)
    denominator = (prior @ drug_program.T).toarray()
    agreement = divide((prior.multiply(np.abs(residual_program)) @ drug_program.T).toarray(), denominator)
    columns = [valid_rna[:, None], divide(measured_counts, drug_counts)[None, :],
               direct, mean_for_drug(np.abs(global_gene)), np.sqrt(np.maximum(0., mean_for_drug(global_gene**2))),
               residual, mean_for_drug(np.abs(residual_gene)), np.sqrt(np.maximum(0., mean_for_drug(residual_gene**2))),
               mean_for_program(global_program), mean_for_program(np.abs(global_program)),
               mean_for_program(residual_program), mean_for_program(np.abs(residual_program)),
               agreement, direct-residual]
    table = np.stack([np.broadcast_to(x, (nr, nd)) for x in columns], axis=-1).astype(np.float32)
    if not np.isfinite(table).all():
        raise ValueError('Nonfinite expression feature')
    return table, dict(global_pathway=global_program.astype(np.float32),
                       within_lineage_pathway=residual_program.astype(np.float32),
                       variable_rna=valid_rna, variable_genes=valid_genes)


def build(args):
    out, ref, uploaded = Path(args.out_dir).resolve(), Path(args.reference_dir).resolve(), Path(args.uploads_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new directory for external inputs')
    previous = base.read_json(ref/'INPUTS.json')
    for name, expected in previous['files'].items():
        if base.digest(ref/name) != expected:
            raise ValueError('Previous frozen annotation changed: '+name)
    out.mkdir(parents=True)
    files = sorted(p for p in uploaded.rglob('*') if p.is_file() and p.suffix in ('.csv', '.zip'))
    source_hashes = {str(p): base.digest(p) for p in files}
    with (ref/'rna_identity.tsv').open() as handle:
        identities = list(csv.DictReader(handle, delimiter='\t'))
    with gzip.open(ref/'sources/hgnc.txt.gz', 'rt') as handle:
        lookup = GeneLookup(list(csv.DictReader(handle, delimiter='\t')))
    dictionaries = base.read_json(ref/'dictionary.json')
    tasks = tuple(dictionaries['rna_names'])
    id_to_rna = {task: defaultdict(list) for task in tasks}
    name_to_rna = {task: defaultdict(set) for task in tasks}
    for row in identities:
        task, idx = row['task'], int(row['rna_node_index_0based'])
        if row['rna_stable_id']:
            id_to_rna[task][row['rna_stable_id']].append(idx)
        for name in (row['rna_name'], row['source_name']):
            if name:
                name_to_rna[task][name.upper()].add(idx)

    # Edge weight never grows with duplicate source rows or numbers of binding sites.
    edges, parse_audit = {}, {}

    def add(task, idx, gene, tier, source, count=1, pmid=''):
        key = (task, int(idx), gene, tier)
        row = edges.setdefault(key, dict(sources=set(), raw_rows=0, pmids=set()))
        row['sources'].add(source)
        row['raw_rows'] += int(count)
        if pmid and pmid not in MISSING:
            row['pmids'].add(numeric_id(pmid))

    for task, name in (('MiDrug', 'encori_mi'), ('LncDrug', 'encori_lnc')):
        counts = Counter()
        path = ref/'sources'/f'{name}.txt.gz'
        source_hashes[str(path)] = base.digest(path)
        with gzip.open(path, 'rt') as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                counts['raw_rows'] += 1
                if task == 'MiDrug':
                    rid, gene = row['miRNAid'], row['geneID']
                    valid = row['geneType'] == 'protein_coding' and int(row['clipExpNum']) >= 1
                else:
                    rid, gene = row['geneID'], row['pairGeneID']
                    valid = (row['geneType'] == 'lncRNA' and row['pairGeneType'] == 'protein_coding'
                             and int(row['interactionNum']) >= 1 and int(row['totalExpNum']) >= 1)
                if not valid or rid not in id_to_rna[task] or not stable_id(gene):
                    continue
                counts['accepted_rows'] += 1
                for idx in id_to_rna[task][rid]:
                    add(task, idx, stable_id(gene), 'binding', name)
        parse_audit[name] = dict(counts)
        print('PARSED', name, dict(counts), flush=True)

    rows, broken = read_lnctard(uploaded/'lnctard/lnctard2.0.zip')
    counts = Counter(raw_rows=len(rows))
    for row in rows:
        tier = lnctard_tier(row)
        counts['source_'+tier] += 1
        if row['RegulatorType'] != 'lncRNA' or row['TargetType'] not in ('PCG', 'TF'):
            counts['wrong_orientation_or_target_type'] += 1
            continue
        resolved, reg_method = lookup.resolve(row['RegulatorEnsembleID'], row['RegulatorEntrezID'], row['Regulator'])
        if reg_method == 'conflicting_or_ambiguous_stable_identifiers':
            counts['regulator_conflicting_identifiers'] += 1
            continue
        rid = stable_id(row['RegulatorEnsembleID'])
        matches = id_to_rna['LncDrug'].get(rid, [])
        if not matches:
            matches = id_to_rna['LncDrug'].get(resolved, [])
            if not matches and not rid and not resolved:
                names = name_to_rna['LncDrug'].get(row['Regulator'].upper(), set())
                matches = list(names) if len(names) == 1 else []
        if not matches:
            counts['regulator_not_resolved_in_project'] += 1
            continue
        gene, method = lookup.resolve(row['TargetEnsembleID'], row['TargetEntrezID'], row['Target'], protein=True)
        counts['target_'+method] += 1
        if not gene:
            continue
        counts['accepted_'+tier+'_rows'] += 1
        for idx in matches:
            add('LncDrug', idx, gene, tier, 'LncTarD_2.0', pmid=row['PubMedID'])
    parse_audit['lnctard'] = dict(counts, invalid_utf8_fields=broken,
                                 classification_field='SearchregulatoryMechanism',
                                 ignored_fields=['Drugs', 'Evidence', 'clincal.application', 'influencedFunction',
                                                 'ExpressionPattern', 'RegulationDiretion'])
    print('PARSED LncTarD', dict(counts), flush=True)

    counts, gene_cache = Counter(), {}
    mti = uploaded/'mirtarbase/hsa_MTI.csv'
    for chunk in pd.read_csv(mti, dtype=str, keep_default_na=False, chunksize=150000):
        counts['raw_rows'] += len(chunk)
        # Filter before the per-edge loop; hsa_MTI also contains viral/nonhuman miRNAs.
        human = chunk['Species (miRNA)'].eq('hsa') & chunk['Species (Target Gene)'].eq('hsa')
        counts['non_human_pair'] += int((~human).sum())
        sub = chunk.loc[human]
        nonfunctional = sub['Support Type'].str.startswith('Non-Functional')
        counts['non_functional'] += int(nonfunctional.sum())
        sub = sub.loc[~nonfunctional].copy()
        sub['project_name'] = sub['miRNA'].str.upper()
        member = sub['project_name'].isin(name_to_rna['MiDrug'])
        counts['not_project_miRNA'] += int((~member).sum())
        sub = sub.loc[member]
        for row in sub.to_dict('records'):
            matches = name_to_rna['MiDrug'][row['project_name']]
            if len(matches) != 1:
                counts['ambiguous_project_miRNA_name'] += 1
                continue
            tier, reason = mti_tier(row['Species (miRNA)'], row['Species (Target Gene)'],
                                    row['Support Type'], row['Experiments'])
            if not tier:
                counts[reason] += 1
                continue
            key = row['Target Gene (Entrez ID)'], row['Target Gene']
            if key not in gene_cache:
                gene_cache[key] = lookup.resolve(entrez=key[0], symbol=key[1], protein=True)
            gene, method = gene_cache[key]
            counts['target_'+method] += 1
            if not gene:
                continue
            counts['accepted_'+tier+'_rows'] += 1
            add('MiDrug', next(iter(matches)), gene, tier, 'miRTarBase_uploaded_human_MTI', pmid=row['References (PMID)'])
    parse_audit['mirtarbase'] = dict(counts)
    print('PARSED miRTarBase', dict(counts), flush=True)

    expression_file = uploaded/'depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv'
    expression = pd.read_csv(expression_file, index_col=0)
    uploaded_expression_shape = expression.shape
    if expression.index.duplicated().any() or expression.columns.duplicated().any():
        raise ValueError('Duplicate expression model IDs or column labels')
    if not np.isfinite(expression.to_numpy(dtype=np.float64)).all():
        raise ValueError('Nonfinite DepMap expression values')
    mapped_columns, column_audit = {}, Counter()
    for column in expression.columns:
        match = re.fullmatch(r'(.+) \((\d+)\)', column)
        if not match:
            raise ValueError('Unrecognized expression gene column: '+column)
        gene, method = lookup.resolve(entrez=match[2], symbol=match[1], protein=True)
        column_audit[method] += 1
        if gene:
            mapped_columns[column] = gene
    model = pd.read_csv(uploaded/'depmap/Model.csv', keep_default_na=False).set_index('ModelID')
    profiles = pd.read_csv(uploaded/'depmap/OmicsProfiles.csv', keep_default_na=False)
    if model.index.duplicated().any():
        raise ValueError('Duplicate Model metadata')
    rna_expression = {}
    for task, name in (('MiDrug', 'MiExp.csv'), ('LncDrug', 'LncExp.csv')):
        frame = pd.read_csv(NET/name, index_col=0)
        frame.columns = [x.upper() for x in frame.columns]
        if frame.index.duplicated().any() or frame.columns.duplicated().any():
            raise ValueError('Duplicate original expression identities')
        rna_expression[task] = frame.loc[:, dictionaries['rna_names'][task]]
        source_hashes[str(NET/name)] = base.digest(NET/name)
    original_cells = list(rna_expression['LncDrug'].index)
    if set(original_cells) != set(rna_expression['MiDrug'].index):
        raise ValueError('Original RNA expression cell sets differ')
    cells = [cell for cell in original_cells if cell in expression.index and cell in model.index]
    if len(cells) < 100:
        raise ValueError('Too few matched cells for expression programs')
    missing = [cell for cell in original_cells if cell not in cells]
    for cell in missing:
        if model['ModelIDAlias'].str.contains(cell, regex=False).any():
            raise ValueError('A missing ACH ID has an official alias; inspect before excluding it')
    expression = expression.loc[cells, list(mapped_columns)].rename(columns=mapped_columns)
    duplicate_mapped = int(expression.columns.duplicated().sum())
    if duplicate_mapped:
        expression = expression.T.groupby(level=0, sort=True).mean().T
    genes = sorted(set(dictionaries['genes']) | {k[2] for k in edges} | set(expression.columns))
    gi = {gene: idx for idx, gene in enumerate(genes)}
    old_gi = [gi[g] for g in dictionaries['genes']]
    old_gp, old_dg = sparse.load_npz(ref/'gene_pathway.npz').tocoo(), sparse.load_npz(ref/'drug_gene.npz').tocoo()
    gp = sparse.csr_matrix((old_gp.data, (np.asarray(old_gi)[old_gp.row], old_gp.col)),
                          shape=(len(genes), old_gp.shape[1]), dtype=np.float64)
    dg = sparse.csr_matrix((old_dg.data, (old_dg.row, np.asarray(old_gi)[old_dg.col])),
                          shape=(old_dg.shape[0], len(genes)), dtype=np.float64)
    matrices = dict(gene_pathway=gp, drug_gene=dg)
    weighted, unions, coverage = {}, {}, {}
    for task in tasks:
        nr = len(dictionaries['rna_names'][task])
        layers = {}
        for tier in TIERS:
            keys = [(idx, gi[g]) for t, idx, g, kind in edges if t == task and kind == tier]
            rr, cc = zip(*keys) if keys else ([], [])
            layers[tier] = sparse.csr_matrix((np.ones(len(keys)), (rr, cc)), shape=(nr, len(genes)))
            matrices[task+'_'+tier] = layers[tier]
        union = binary(sum(layers.values()))
        combined = sparse.csr_matrix(union.shape, dtype=np.float64)
        for tier in TIERS:
            combined = combined.maximum(layers[tier]*WEIGHTS[tier])
        matrices[task+'_union'], matrices[task+'_weighted'] = union, combined
        unions[task], weighted[task] = union, combined
        coverage[task] = dict(total=nr,
            tiers={tier:dict(covered_rnas=int((np.diff(m.indptr)>0).sum()), edges=m.nnz)
                   for tier, m in layers.items()},
            union_rnas=int((np.diff(union.indptr)>0).sum()), union_edges=union.nnz,
            pathway_rnas=int((np.diff((union@gp).tocsr().indptr)>0).sum()))

    gx = np.zeros((len(cells), len(genes)), dtype=np.float64)
    gx[:, [gi[g] for g in expression.columns]] = expression.to_numpy(dtype=np.float64)
    groups = model.loc[cells, 'OncotreeLineage'].replace('', 'Unknown').to_numpy(dtype=str)
    projection = program_projection(gp)
    programs = {}
    for task in tasks:
        values = rna_expression[task].loc[cells].to_numpy(dtype=np.float64)
        table, detail = contextual_features(values, gx, groups, dg, projection, weighted[task])
        np.save(out/(task+'_expression_features.npy'), table)
        programs.update({task+'_'+key:value for key,value in detail.items()})
        coverage[task]['variable_expression_rnas'] = int(detail['variable_rna'].sum())
        print('EXPRESSION_FEATURES', task, table.shape, flush=True)
    np.savez_compressed(out/'expression_programs.npz', **programs)
    for name, matrix in matrices.items():
        sparse.save_npz(out/(name+'.npz'), matrix)
    with gzip.open(out/'rna_gene_evidence.tsv.gz', 'wt') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['task','rna_index','rna_name','gene_id','tier','sources','raw_rows','PMIDs'])
        for (task, idx, gene, tier), row in sorted(edges.items()):
            writer.writerow([task,idx,dictionaries['rna_names'][task][idx],gene,tier,
                             '|'.join(sorted(row['sources'])),row['raw_rows'],'|'.join(sorted(row['pmids']))])
    dictionaries['genes'] = genes
    base.write_json(out/'dictionary.json', dictionaries)
    (out/'ACH_ids.txt').write_text('\n'.join(cells)+'\n')
    base.write_json(out/'cell_lineages.json', dict(zip(cells, groups)))
    shutil.copy2(ref/'rna_identity.tsv', out/'rna_identity.tsv')
    coverage['Drug'] = dict(total=dg.shape[0], gene_annotated=int((np.diff(dg.indptr)>0).sum()),
                            pathway_annotated=int((np.diff((dg@gp).tocsr().indptr)>0).sum()), edges=dg.nnz)
    available_rna_profiles = set(profiles.loc[profiles.DataType.eq('rna'), 'ModelID'])
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), coverage=coverage,
        parse_audit=parse_audit, gene_count=len(genes), pathway_count=gp.shape[1], reactome_version='97',
        depmap=dict(release=args.depmap_release, release_evidence='User confirmed all three files are 25Q2',
                    original_cells=len(original_cells), matched_cells=len(cells), missing_cells=missing,
                    expression_models=uploaded_expression_shape[0],
                    uploaded_gene_columns=sum(column_audit.values()), mapped_expression_genes=len(expression.columns),
                    gene_mapping=dict(column_audit), duplicate_gene_columns_averaged=duplicate_mapped,
                    expression_models_without_rna_profile=[c for c in cells if c not in available_rna_profiles],
                    lineages=dict(Counter(groups)), alignment='Exact ACH/ModelID; both RNA matrices reindexed by ID.',
                    matrix_semantics='Uploaded 25Q2 protein-coding TPM logp1, one row per model. Rank correlations are invariant to monotone log transformation.'),
        tier_weights=WEIGHTS, expression_feature_names=list(CONTEXT_NAMES),
        preprocessing='Across-cell average-tie ranks, centered and L2 normalized; second view also centers ranks within OncotreeLineage. Pathway scores are normalized weighted sums of gene ranks, not experimentally measured activities.',
        encori_lower_support_incomplete_for=['NEAT1 below support 2','MALAT1 below support 3'],
        ignored_uploads=['miRTarBase_MTI.csv: overlapping multi-species table; not added twice',
                         'MicroRNA_Target_Sites.csv: binding positions not needed for this first feature trial',
                         'GEOdiffexpression.zip','CTCdiffexpression.zip','TCGAdiffexpression.zip'],
        labels_read=False, RNA_drug_edges_read=False,
        drug_annotation_fields_used=['frozen ChEMBL/UniProt drug_gene matrix'],
        limitations=['Curated functional interactions can be indirect and context-specific.',
                     'Expression associations and CLIP evidence do not establish regulatory causality.',
                     'Original RNA-expression release is not known; matched 25Q2 mRNA data may differ in processing.',
                     'Uploaded miRTarBase exact release not separately verified; original bytes identify this input snapshot.',
                     'Entity attributes are transductive and not a temporal evidence holdout.'])
    base.write_json(out/'COVERAGE.json', report)
    shutil.copy2(__file__, out/'BUILD_SOURCE.py')
    source_hashes[str(ref/'INPUTS.json')] = base.digest(ref/'INPUTS.json')
    source_hashes[str(ref/'sources/hgnc.txt.gz')] = base.digest(ref/'sources/hgnc.txt.gz')
    source_hashes[str(ref/'rna_identity.tsv')] = base.digest(ref/'rna_identity.tsv')
    base.write_json(out/'INPUTS.json', dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        files={p.name:base.digest(p) for p in sorted(out.iterdir()) if p.is_file()},
        external_sources=source_hashes, previous_inputs_sha256=base.digest(ref/'INPUTS.json'),
        source_code_sha256=base.digest(__file__), labels_read=False))
    print('INPUTS_FROZEN', out, json.dumps(coverage), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default=str(DEFAULT_OUT))
    parser.add_argument('--reference-dir', default=str(REFERENCE))
    parser.add_argument('--uploads-dir', default=str(UPLOADS))
    parser.add_argument('--depmap-release', default='25Q2')
    build(parser.parse_args())


if __name__ == '__main__':
    main()
