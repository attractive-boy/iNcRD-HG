"""Audit molecular/gene identities only; never open RNA-drug association outcomes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase, RDLogger
from scipy import sparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NET = ROOT.parents[1] / 'incrd-run/2.construct_network'
CONTEXT = ROOT / 'results/tabicl_pathway_context_inputs_20260913'
OUT = HERE / 'MORPHOLOGY_IDENTITY_COVERAGE_V1.json'
if OUT.exists():
    raise ValueError('Preserve identity audit')
RDLogger.DisableLog('rdApp.*')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    if not value:
        return ''
    molecule = Chem.MolFromSmiles(value)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True) if molecule is not None else ''


manifest = json.loads((HERE / 'JUMP_METADATA_RETRY_V2.json').read_text())
for record in manifest['records']:
    if not record.get('git_blob_verified') or digest(record['file']) != record['sha256']:
        raise ValueError('Metadata incomplete or changed')
dictionary = json.loads((CONTEXT / 'dictionary.json').read_text())
drugs = pd.read_csv(NET / 'Drug.csv', dtype=str, keep_default_na=False).set_index('Drug_Name')
smiles = pd.read_csv(NET / 'drug_smiles.csv', dtype=str, keep_default_na=False).set_index('drug_cid')
compounds = pd.read_csv(HERE / 'jump_v2_metadata_compound.csv.gz', dtype=str, keep_default_na=False)
if compounds.Metadata_JCP2022.duplicated().any():
    raise ValueError('Duplicate JUMP compound identity')
canonical_jump = [canonical(value) for value in compounds.Metadata_SMILES]
lookup = {}
for i, value in enumerate(canonical_jump):
    if value:
        lookup.setdefault(value, []).append(i)
drug_rows = []
for i, name in enumerate(dictionary['drug_names']):
    cid = drugs.loc[name, 'CID']
    value = canonical(smiles.loc[cid, 'smiles'])
    matches = lookup.get(value, []) if value else []
    drug_rows.append(dict(drug_index=i, drug_name=name, cid=cid, canonical_isomeric_smiles=value,
        count=len(matches), jcp2022='|'.join(compounds.iloc[matches].Metadata_JCP2022.tolist())))
pd.DataFrame(drug_rows).to_csv(HERE / 'drug_identity_matches_v1.csv', index=False)
crispr = pd.read_csv(HERE / 'jump_v2_metadata_crispr.csv.gz', dtype=str, keep_default_na=False)
hgnc = ROOT / 'results/tabicl_pathway_inputs_20260913/sources/hgnc.txt.gz'
symbols = pd.read_csv(hgnc, sep='\t', dtype=str, keep_default_na=False)
entrez = {}
for row in symbols.loc[symbols.status.eq('Approved')].to_dict('records'):
    if row['entrez_id'] and row['ensembl_gene_id']:
        entrez.setdefault(row['entrez_id'], set()).add(row['ensembl_gene_id'])
genes = {g: i for i, g in enumerate(dictionary['genes'])}
gene_rows = []
for row in crispr.to_dict('records'):
    choices = entrez.get(row['Metadata_NCBI_Gene_ID'], set())
    ens = next(iter(choices)) if len(choices) == 1 else ''
    gene_rows.append(dict(jcp2022=row['Metadata_JCP2022'], entrez=row['Metadata_NCBI_Gene_ID'],
        symbol=row['Metadata_Symbol'], ensembl=ens, dictionary_index=genes.get(ens, -1),
        ambiguous=len(choices) > 1))
pd.DataFrame(gene_rows).to_csv(HERE / 'gene_identity_matches_v1.csv', index=False)
matched_gene = sorted({r['dictionary_index'] for r in gene_rows if r['dictionary_index'] >= 0})
rna_coverage = {}
files = [CONTEXT / 'dictionary.json', NET / 'Drug.csv', NET / 'drug_smiles.csv', hgnc,
         HERE / 'JUMP_METADATA_RETRY_V2.json', *[Path(r['file']) for r in manifest['records']]]
for tier in ['functional', 'binding', 'association', 'union']:
    path = CONTEXT / ('MiDrug_' + tier + '.npz')
    matrix = sparse.load_npz(path).tocsr()
    selected = matrix[:, matched_gene]
    counts = np.asarray((selected != 0).sum(1)).ravel()
    rna_coverage[tier] = dict(rnas=matrix.shape[0], total_edges=int(matrix.nnz),
        retained_edges=int(selected.nnz), rnas_with_at_least_one=int((counts > 0).sum()),
        target_count_quantiles=np.quantile(counts, [0, .25, .5, .75, 1]).tolist())
    files.append(path)
result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), association_outcomes_read=False,
    source_sha256=digest(__file__), input_sha256={str(p): digest(p) for p in files}, rdkit_version=rdBase.rdkitVersion,
    identity_policy='Exact canonical isomeric SMILES; preserve salts, stereochemistry, charges and fragments. HGNC approved Entrez-to-Ensembl unique mappings only.',
    jump_compounds=len(compounds), jump_invalid_smiles=sum(not x for x in canonical_jump),
    project_drugs=len(drug_rows), matched_project_drugs=sum(r['count'] > 0 for r in drug_rows),
    exact_matched_jcp_compounds=len({j for r in drug_rows for j in r['jcp2022'].split('|') if j}),
    crispr_entries=len(crispr), matched_crispr_entries=sum(r['dictionary_index'] >= 0 for r in gene_rows),
    unique_dictionary_genes=len(matched_gene), rna_coverage=rna_coverage,
    available_profile_rows_not_yet_verified=True, neural_fits=0,
    files={name: digest(HERE / name) for name in ['drug_identity_matches_v1.csv', 'gene_identity_matches_v1.csv']})
OUT.write_text(json.dumps(result, indent=2) + '\n')
for key in ['jump_compounds', 'jump_invalid_smiles', 'project_drugs', 'matched_project_drugs',
            'exact_matched_jcp_compounds', 'crispr_entries', 'matched_crispr_entries', 'unique_dictionary_genes', 'rna_coverage']:
    print(key, result[key], flush=True)
