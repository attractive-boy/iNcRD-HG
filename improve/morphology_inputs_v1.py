"""Freeze measured drug/CRISPR morphology and external functional identities.

Run with the existing CPU preprocessing environment that provides RDKit and
PyArrow. No RNA-drug association arrays or project evaluation outcomes are read.
"""
from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
from rdkit import Chem, rdBase, RDLogger
from scipy import sparse

ROOT = Path(__file__).resolve().parent
HERE = ROOT / 'research/morphology_next_20260916'
CONTEXT = ROOT / 'results/tabicl_pathway_context_inputs_20260913'
OUT = ROOT / 'results/morphology_inputs_v1_20260916'
RDLogger.DisableLog('rdApp.*')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    if not isinstance(value, str) or not value:
        return ''
    molecule = Chem.MolFromSmiles(value)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True) if molecule is not None else ''


def seal(value):
    sha = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(value, sha256=sha)


def aggregate(source, controls):
    if not source.get('s3_etag_verified') or digest(source['file']) != source['sha256']:
        raise ValueError('Unverified profile object')
    frame = pq.read_table(source['file'], memory_map=True).to_pandas()
    columns = [name for name in frame.columns if name.startswith('X_')]
    if columns != ['X_' + str(i + 1) for i in range(len(columns))]:
        raise ValueError('Unexpected anonymous feature ordering')
    values = frame[columns].to_numpy(copy=False)
    finite = np.isfinite(values).all(1)
    noncontrol = ~frame.Metadata_JCP2022.astype(str).isin(controls).to_numpy()
    retained = finite & noncontrol
    report = dict(raw_rows=len(frame), feature_count=len(columns), nonfinite_rows=int((~finite).sum()),
                  control_rows=int((~noncontrol).sum()), retained_rows=int(retained.sum()))
    frame = frame.loc[retained, ['Metadata_JCP2022', 'Metadata_Source'] + columns]
    if frame.Metadata_JCP2022.isna().any() or frame.Metadata_Source.isna().any():
        raise ValueError('Missing perturbation/source identity')
    frame['Metadata_JCP2022'] = frame.Metadata_JCP2022.astype(str)
    frame['Metadata_Source'] = frame.Metadata_Source.astype(str)
    counts = frame.groupby(['Metadata_JCP2022', 'Metadata_Source']).size().rename('well_count')
    counts.to_csv(OUT / (source['name'] + '_well_counts.csv.gz'))
    # Each laboratory/source receives equal weight for a perturbation after
    # averaging its measured wells. Do not let a more replicated lab dominate.
    grouped = frame.groupby(['Metadata_JCP2022', 'Metadata_Source'])[columns].mean()
    means = grouped.groupby(level='Metadata_JCP2022').mean()
    result = means.to_numpy(dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite aggregate')
    report.update(perturbations=len(means), source_perturbation_groups=len(grouped),
        sources=sorted(frame.Metadata_Source.unique().tolist()),
        raw_absolute_quantiles=np.quantile(np.abs(result), [.5, .95, .99, 1]).tolist(),
        aggregation='Mean wells within JCP2022/source, then equal mean over sources. No dose or cell-line equivalence is inferred from these tables.')
    keys = means.index.astype(str).to_numpy()
    del frame, values, grouped, means
    gc.collect()
    print('MORPHOLOGY_AGGREGATED', source['name'], result.shape, flush=True)
    return keys, result, report


def normalize(values):
    mean = values.astype(np.float64).mean(0)
    sd = values.astype(np.float64).std(0)
    scale = np.maximum(sd, 1e-6)
    unbounded = (values.astype(np.float64) - mean) / scale
    result = np.clip(unbounded, -10, 10).astype(np.float32)
    return result, mean, scale, dict(clipped_values=int((np.abs(unbounded) > 10).sum()),
        total_values=int(values.size), below_sd_floor=int((sd < 1e-6).sum()),
        scope='All noncontrol externally measured JCP2022 aggregate profiles in this modality; no project association outcomes.')


def main():
    if OUT.exists():
        raise ValueError('Preserve morphology inputs; a rebuild needs a new version')
    download = json.loads((HERE / 'PROFILE_DOWNLOAD_V1.json').read_text())
    if not download['all_complete']:
        raise ValueError('Profile download incomplete')
    identity = json.loads((HERE / 'MORPHOLOGY_IDENTITY_COVERAGE_V1.json').read_text())
    for name, expected in identity['files'].items():
        if digest(HERE / name) != expected:
            raise ValueError('Identity mapping changed')
    context = json.loads((CONTEXT / 'INPUTS.json').read_text())
    if context['labels_read']:
        raise ValueError('Context evidence read project labels')
    for name in ['dictionary.json', 'MiDrug_functional.npz']:
        if digest(CONTEXT / name) != context['files'][name]:
            raise ValueError('Functional identity matrix changed')
    external_manifest = json.loads((HERE / 'AUTHOR_DTI_TRAIN_FETCH_V1.json').read_text())
    if not external_manifest['git_blob_verified'] or digest(external_manifest['file']) != external_manifest['sha256']:
        raise ValueError('External DTI training table changed')
    OUT.mkdir(parents=True)
    controls_frame = pd.read_csv(HERE / 'jump_v2_metadata_perturbation_control.csv', dtype=str, keep_default_na=False)
    negative = controls_frame.loc[controls_frame.Metadata_pert_type.str.lower().isin(['negcon', 'empty'])]
    controls = set(negative.Metadata_JCP2022) | set(negative.Metadata_Name) | {'DMSO', 'EMPTY', 'no-guide', 'non-targeting'}
    modality, aggregate_reports, normalization = {}, {}, {}
    for source in download['files']:
        keys, values, report = aggregate(source, controls)
        values, mean, scale, normal = normalize(values)
        modality[source['name']] = (keys, values)
        aggregate_reports[source['name']] = report
        normalization[source['name']] = normal
        np.savez_compressed(OUT / (source['name'] + '_normalization.npz'), mean=mean, scale=scale)
    compounds = pd.read_csv(HERE / 'jump_v2_metadata_compound.csv.gz', dtype=str, keep_default_na=False)
    compound_keys, compound_values = modality['profile_compound']
    measured = {key: i for i, key in enumerate(compound_keys)}
    chemical = defaultdict(list)
    canonical_rows = []
    for row in compounds.to_dict('records'):
        value = canonical(row['Metadata_SMILES'])
        canonical_rows.append(dict(jcp2022=row['Metadata_JCP2022'], canonical_isomeric_smiles=value,
                                   measured=row['Metadata_JCP2022'] in measured))
        if value and row['Metadata_JCP2022'] in measured:
            chemical[value].append(row['Metadata_JCP2022'])
    pd.DataFrame(canonical_rows).to_csv(OUT / 'compound_canonical_identities.csv.gz', index=False)
    gene_keys, gene_values = modality['profile_crispr']
    measured_genes = {key: i for i, key in enumerate(gene_keys)}
    gene_frame = pd.read_csv(HERE / 'gene_identity_matches_v1.csv', keep_default_na=False)
    gene_groups = defaultdict(list)
    gene_symbols = defaultdict(set)
    for row in gene_frame.to_dict('records'):
        if row['dictionary_index'] >= 0 and row['jcp2022'] in measured_genes:
            gene_groups[row['ensembl']].append(row['jcp2022'])
            gene_symbols[row['symbol']].add(row['ensembl'])
    genes = sorted(gene_groups)
    gene_index = {key: i for i, key in enumerate(genes)}
    symbols = {symbol: next(iter(choices)) for symbol, choices in gene_symbols.items() if len(choices) == 1}
    genes_x = np.stack([gene_values[[measured_genes[jcp] for jcp in gene_groups[g]]].mean(0) for g in genes])
    external = pd.read_csv(external_manifest['file'], dtype=str, keep_default_na=False)
    if not {'SMILES', 'target'} <= set(external.columns):
        raise ValueError('External DTI identifiers absent')
    unique_smiles = {value: canonical(value) for value in external.SMILES.unique()}
    external_rows, positive = [], set()
    for position, row in enumerate(external.to_dict('records')):
        key, gene = unique_smiles[row['SMILES']], symbols.get(row['target'], '')
        valid_drug, valid_gene = key in chemical, gene in gene_index
        external_rows.append(dict(source_row=position, valid_drug=valid_drug, valid_gene=valid_gene,
                                  canonical_smiles=key, target_symbol=row['target'], ensembl=gene))
        if valid_drug and valid_gene:
            positive.add((key, gene))
    if len(positive) < 100:
        raise ValueError('Insufficient measured external DTI alignment')
    project = pd.read_csv(HERE / 'drug_identity_matches_v1.csv', keep_default_na=False)
    drug_keys = sorted({key for key, gene in positive} | {key for key in project.canonical_isomeric_smiles if key in chemical})
    drug_index = {key: i for i, key in enumerate(drug_keys)}
    drug_x = np.stack([compound_values[[measured[jcp] for jcp in chemical[key]]].mean(0) for key in drug_keys])
    project_index = np.array([drug_index.get(key, -1) for key in project.canonical_isomeric_smiles], dtype=np.int64)
    dictionary = json.loads((CONTEXT / 'dictionary.json').read_text())
    if dictionary['drug_names'] != project.drug_name.tolist():
        raise ValueError('Project drug order changed')
    old_gene = {key: i for i, key in enumerate(dictionary['genes'])}
    rna = sparse.load_npz(CONTEXT / 'MiDrug_functional.npz').tocsr()[:, [old_gene[g] for g in genes]]
    rna = (rna.toarray() != 0).astype(np.float32)
    counts = rna.sum(1)
    rna /= np.maximum(counts[:, None], 1)
    permutation = np.random.default_rng(32016).permutation(len(genes))
    dti = np.array(sorted({(drug_index[key], gene_index[g]) for key, g in positive}), dtype=np.int64).T
    np.savez_compressed(OUT / 'arrays.npz', drug_profile=drug_x, gene_profile=genes_x,
        rna_gene=rna, rna_gene_counts=counts, rna_permutation=permutation,
        project_drug_index=project_index, dti_pairs=dti, drug_keys=np.array(drug_keys), gene_ids=np.array(genes),
        drug_names=np.array(dictionary['drug_names']), rna_names=np.array(dictionary['rna_names']['MiDrug']))
    pd.DataFrame(external_rows).to_csv(OUT / 'external_dti_mapping.csv.gz', index=False)
    pd.DataFrame([dict(drug_index=i, canonical_smiles=key, jcp2022='|'.join(chemical[key]))
                  for key, i in drug_index.items()]).to_csv(OUT / 'training_drug_dictionary.csv.gz', index=False)
    pd.DataFrame([dict(gene_index=i, ensembl=g, jcp2022='|'.join(gene_groups[g]))
                  for g, i in gene_index.items()]).to_csv(OUT / 'training_gene_dictionary.csv', index=False)
    inputs = [HERE / name for name in ['PROFILE_DOWNLOAD_V1.json', 'MORPHOLOGY_IDENTITY_COVERAGE_V1.json',
        'AUTHOR_DTI_TRAIN_FETCH_V1.json', 'jump_v2_metadata_compound.csv.gz', 'jump_v2_metadata_crispr.csv.gz',
        'jump_v2_metadata_perturbation_control.csv', 'drug_identity_matches_v1.csv', 'gene_identity_matches_v1.csv']]
    inputs += [CONTEXT / name for name in ['INPUTS.json', 'dictionary.json', 'MiDrug_functional.npz']]
    inputs.append(Path(external_manifest['file']))
    result = dict(utc=datetime.now(timezone.utc).isoformat(), source_sha256=digest(__file__),
        input_sha256={str(p): digest(p) for p in inputs},
        source_profile_sha256={row['file']: row['sha256'] for row in download['files']},
        software=dict(rdkit=rdBase.rdkitVersion, pyarrow=pyarrow.__version__, pandas=pd.__version__, numpy=np.__version__),
        aggregate_reports=aggregate_reports, normalization=normalization,
        identity_policy=identity['identity_policy'], project_drugs=len(project_index),
        measured_project_drugs=int((project_index >= 0).sum()), project_rnas=len(rna),
        rnas_with_functional_targets=int((counts > 0).sum()), functional_target_edges=int((rna != 0).sum()),
        gene_profiles=len(genes), training_drug_profiles=len(drug_keys), external_raw_dti_rows=len(external),
        external_unique_aligned_dti_pairs=dti.shape[1], external_alignment_drugs=len(np.unique(dti[0])),
        external_alignment_genes=len(np.unique(dti[1])),
        external_dti_semantics='Author-released positive image-DTI training table; author describes direct binding curation. Individual source assays not independently reverified. No sign, resistance effect, or measured ncRNA intervention is inferred.',
        rna_semantics='Existing binary functional-evidence gene sets; mean mapped CRISPR embedding is a learned proxy, not a measured miRNA knockdown profile.',
        author_weights_reused=False, author_599d_crispr_checkpoint_incompatible_with_current_259d_profiles=True,
        contrast_control='One fixed global permutation of mapped gene identities, seed 32016, preserves every RNA target count and all pairwise target-set overlaps; it preserves the degree multiset, not each named gene degree.',
        association_outcomes_read=False, new_neural_fits=0,
        files={p.name: digest(p) for p in OUT.iterdir() if p.is_file()})
    (OUT / 'INPUTS.json').write_text(json.dumps(seal(result), indent=2) + '\n')
    for key in ['measured_project_drugs', 'rnas_with_functional_targets', 'functional_target_edges', 'gene_profiles',
                'training_drug_profiles', 'external_raw_dti_rows', 'external_unique_aligned_dti_pairs', 'external_alignment_drugs', 'external_alignment_genes']:
        print(key, result[key], flush=True)
    print('MORPHOLOGY_INPUTS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
