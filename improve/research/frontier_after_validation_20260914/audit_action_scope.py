"""Label-free follow-up: action granularity and molecular-input readiness."""
from collections import Counter
from datetime import datetime, timezone
import json

import numpy as np
from scipy import sparse

from audit_existing_information import (OUT, REFERENCE, MECHANISMS, WINDOWS,
    SOURCE_FILTER, WORKSPACE, digest, tsv, pmids, support_class)


def main():
    output = OUT / 'ACTION_SCOPE_AUDIT.json'
    if output.exists():
        raise ValueError('Refuse overwrite')
    dictionary_path = REFERENCE / 'dictionary.json'
    dictionary = json.loads(dictionary_path.read_text())
    gene_index = {g: i for i, g in enumerate(dictionary['genes'])}
    n_rna, n_drug = len(dictionary['rna_names']['MiDrug']), len(dictionary['drug_names'])
    mechanisms = tsv(MECHANISMS)
    sites = tsv(WINDOWS / 'sites.tsv')
    forbidden = set(json.loads(SOURCE_FILTER.read_text())['forbidden_pmids'])
    sites = [r for r in sites if pmids(r['pmids']) and not pmids(r['pmids']) & forbidden]
    results = {}
    for target_type in sorted({r['target_type'] for r in mechanisms}):
        rows = [r for r in mechanisms if r['target_type'] == target_type]
        edges = {(int(r['drug_node_index_0based']), gene_index[r['gene_id']]) for r in rows}
        drug_matrix = sparse.csr_matrix((np.ones(len(edges), dtype=np.int32),
             ([d for d, g in edges], [g for d, g in edges])),
             shape=(n_drug, len(gene_index)))
        counts = {}
        for label, subset in [('all_source_filtered_sites', sites),
                ('strong_functional_no_nonfunctional', [r for r in sites if support_class(r['support_types']) == 'strong_functional_no_nonfunctional'])]:
            site_matrix = sparse.csr_matrix((np.ones(len(subset), dtype=np.int32),
                 ([int(r['rna_index']) for r in subset], [gene_index[r['gene_id']] for r in subset])),
                 shape=(n_rna, len(gene_index)))
            pairs = (site_matrix @ drug_matrix.T).tocsr()
            counts[label] = dict(pairs=int(pairs.nnz), fraction_of_full_dictionary=pairs.nnz / (n_rna * n_drug),
                rnas=int(np.count_nonzero(pairs.getnnz(1))), drugs=int(np.count_nonzero(pairs.getnnz(0))))
        results[target_type] = dict(mechanism_rows=len(rows), unique_drug_gene_edges=len(edges),
            drugs=len({d for d, g in edges}), genes=len({g for d, g in edges}),
            actions_by_row=dict(Counter(r['action'] for r in rows)), site_pair_coverage=counts)
    smiles_path = WORKSPACE / 'incrd-run/2.construct_network/drug_smiles.csv'
    import csv
    with smiles_path.open(encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        drug_rows = list(reader)
    drug_entity_path = smiles_path.with_name('Drug.csv')
    with drug_entity_path.open(encoding='utf-8-sig') as handle:
        entities = list(csv.DictReader(handle))
    column = next(k for k in reader.fieldnames if 'smiles' in k.lower())
    smiles_summary = dict(rows=len(drug_rows), smiles_column=column,
        nonempty_smiles=sum(bool(r[column]) for r in drug_rows),
        names_match_dictionary=[r['Drug_Name'] for r in entities] == dictionary['drug_names'],
        cid_sets_match=set(r['drug_cid'] for r in drug_rows) == set(r['CID'] for r in entities),
        unique_smiles_cids=len({r['drug_cid'] for r in drug_rows}),
        new_encoders_executed=False, new_checkpoint_downloaded=False)
    try:
        from rdkit import Chem, rdBase
        molecules = [Chem.MolFromSmiles(r[column]) for r in drug_rows]
        smiles_summary.update(rdkit_version=rdBase.rdkitVersion,
            valid_molecules=sum(m is not None for m in molecules),
            multicomponent_molecules=sum(len(Chem.GetMolFrags(m)) > 1 for m in molecules if m is not None),
            limitation='RDKit parse success does not establish compatibility with M2UMol/SELFormerMM preprocessing or checkpoint availability.')
    except ImportError:
        smiles_summary['rdkit_validation'] = 'not_available_in_current_environment'
    inputs = [dictionary_path, MECHANISMS, WINDOWS / 'sites.tsv', SOURCE_FILTER, smiles_path, drug_entity_path,
              OUT / 'audit_existing_information.py', OUT / 'audit_action_scope.py']
    result = dict(as_of='2026-09-14', completed_at_utc=datetime.now(timezone.utc).isoformat(),
        outcome_labels_read=False, coverage_denominator=n_rna*n_drug,
        inputs=[dict(path=str(p), sha256=digest(p)) for p in inputs],
        target_type_scopes=results, molecular_input_readiness=smiles_summary,
        interpretation='Complex/family mechanisms are not automatically per-component gene perturbation signs. Scopes can overlap and are not additive.')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(single_protein=results['SINGLE PROTEIN'], smiles=smiles_summary), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
