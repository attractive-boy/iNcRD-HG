"""Audit external feature information without loading any outcome labels.

Coverage is over the full dictionary Cartesian product, never a scored split.
Only the forbidden PMID metadata in SOURCE_FILTER.json is used; no original
records, source labels, test pairs, predictions or holdout arrays are opened.
"""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy import sparse


OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
WORKSPACE = REPO.parent
REFERENCE = REPO / 'improve/results/tabicl_pathway_context_inputs_20260913'
MECHANISMS = REPO / 'improve/results/tabicl_pathway_inputs_20260913/drug_to_gene.tsv'
EXPANDED = WORKSPACE / 'data_external/functional_priority_20260913/chembl'
WINDOWS = REPO / 'improve/results/rna_functional_inputs_20260914_v2/windows'
SOURCE_FILTER = REPO / 'improve/results/context_functional_20260914/SOURCE_FILTER.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tsv(path):
    with path.open(encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def support_class(value):
    tokens = set(value.split(';'))
    known = {'Functional MTI', 'Functional MTI (Weak)',
             'Non-Functional MTI', 'Non-Functional MTI (Weak)'}
    if not tokens or not tokens <= known:
        raise ValueError('Unknown support category: ' + value)
    functional = bool(tokens & {'Functional MTI', 'Functional MTI (Weak)'})
    nonfunctional = bool(tokens & {'Non-Functional MTI', 'Non-Functional MTI (Weak)'})
    if functional and nonfunctional:
        return 'mixed_functional_nonfunctional'
    if nonfunctional:
        return 'nonfunctional_only'
    if 'Functional MTI' in tokens:
        return 'strong_functional_no_nonfunctional'
    return 'weak_functional_only'


def pmids(value):
    # Match the frozen context_functional_eval.auxiliary_pmids semantics.
    tokens = [re.sub(r'^([1-9][0-9]*)\.0$', r'\1', x)
              for x in re.split(r'[;,|\s]+', value) if x]
    if any(not re.fullmatch(r'[1-9][0-9]*', x) for x in tokens):
        raise ValueError('Malformed auxiliary PMID')
    return set(tokens)


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    output = OUT / 'LOCAL_INFORMATION_AUDIT.json'
    if output.exists():
        raise ValueError('Audit already archived; do not overwrite')
    inputs = [REFERENCE / 'dictionary.json', REFERENCE / 'drug_gene.npz',
              MECHANISMS, REFERENCE / 'INPUTS.json',
              EXPANDED / 'drug_gene_expanded.npz', EXPANDED / 'INPUTS.json',
              WINDOWS / 'sites.tsv', SOURCE_FILTER,
              REPO / 'improve/rna_functional_inputs.py',
              REPO / 'improve/tabicl_functional_modules.py',
              REPO / 'improve/context_functional_eval.py',
              REPO / 'improve/context_source_eval.py', Path(__file__)]
    dictionary = json.loads(inputs[0].read_text())
    genes = {gene: i for i, gene in enumerate(dictionary['genes'])}
    n_rna, n_drug = len(dictionary['rna_names']['MiDrug']), len(dictionary['drug_names'])
    matrices = {'original_mechanism': sparse.load_npz(inputs[1]).tocsr(),
                'expanded_binding': sparse.load_npz(inputs[4]).tocsr()}
    for directory, file in [(REFERENCE, inputs[0]), (REFERENCE, inputs[1]),
                            (EXPANDED, inputs[4])]:
        manifest = json.loads((directory / 'INPUTS.json').read_text())
        if manifest['files'][file.name] != digest(file):
            raise ValueError('Frozen input hash mismatch: ' + str(file))
    expansion_manifest = json.loads((EXPANDED / 'INPUTS.json').read_text())
    assert expansion_manifest['dictionary_sha256'] == digest(REFERENCE / 'dictionary.json')
    assert expansion_manifest['reference_manifest_sha256'] == digest(REFERENCE / 'INPUTS.json')
    for mat in matrices.values():
        assert mat.shape == (n_drug, len(genes))
        assert np.all(mat.data == 1)
    assert matrices['original_mechanism'].minimum(matrices['expanded_binding']).nnz == matrices['original_mechanism'].nnz
    rows = tsv(WINDOWS / 'sites.tsv')
    assert len({row['site_id'] for row in rows}) == len(rows)
    for row in rows:
        assert row['rna_name'] == dictionary['rna_names']['MiDrug'][int(row['rna_index'])]
        assert row['gene_id'] in genes
        assert int(row['window_length']) == len(row['target_sequence']) == 50
    source_metadata = json.loads(SOURCE_FILTER.read_text())
    forbidden = set(source_metadata['forbidden_pmids'])
    kept, removed, removal = [], [], Counter()
    for row in rows:
        references = pmids(row['pmids'])
        if not references:
            removal['missing_source'] += 1
            removed.append(row)
        elif references & forbidden:
            removal['overlap_with_validation_or_holdout'] += 1
            removed.append(row)
        else:
            kept.append(row)
    removal['retained_sites'] = len(kept)
    assert dict(removal) == source_metadata['filter_stats']
    assert not {p for row in kept for p in pmids(row['pmids'])} & forbidden
    groups = {'all_cached_windows': rows, 'source_filtered_windows': kept}
    summaries, drug_rows = {}, []
    for name, selected in groups.items():
        classes = Counter(support_class(row['support_types']) for row in selected)
        subsets = {'all': selected}
        subsets.update({label: [row for row in selected if support_class(row['support_types']) == label]
                        for label in sorted(classes)})
        summary = {'windows': len(selected),
                   'rna_count': len({row['rna_index'] for row in selected}),
                   'gene_count': len({row['gene_id'] for row in selected}),
                   'unique_rna_gene_edges': len({(row['rna_index'], row['gene_id']) for row in selected}),
                   'raw_support_counts': dict(sorted(Counter(row['support_types'] for row in selected).items())),
                   'support_classes': dict(sorted(classes.items())), 'coverage': {}}
        for label, subset in subsets.items():
            site_matrix = sparse.csr_matrix((np.ones(len(subset), dtype=np.int32),
                ([int(row['rna_index']) for row in subset], [genes[row['gene_id']] for row in subset])),
                shape=(n_rna, len(genes)))
            summary['coverage'][label] = {}
            for matrix_name, drug_matrix in matrices.items():
                pair_sites = (site_matrix @ drug_matrix.T).tocsr()
                pair_sites.eliminate_zeros()
                pair_identity = pair_sites.copy()
                pair_identity.data[:] = 1
                rna_counts = np.asarray(pair_identity.sum(axis=1)).ravel()
                drug_counts = np.asarray(pair_identity.sum(axis=0)).ravel()
                active_genes = np.flatnonzero(np.asarray(site_matrix.sum(axis=0)).ravel())
                linked_genes = np.intersect1d(active_genes, np.unique(drug_matrix.indices))
                summary['coverage'][label][matrix_name] = {
                    'possible_pairs': n_rna * n_drug,
                    'pairs_with_at_least_one_exact_gene_match': int(pair_sites.nnz),
                    'pair_fraction': pair_sites.nnz / (n_rna * n_drug),
                    'rnas_with_any_matching_drug': int(np.count_nonzero(rna_counts)),
                    'drugs_with_any_matching_rna': int(np.count_nonzero(drug_counts)),
                    'shared_genes': len(linked_genes),
                    'site_drug_links_with_site_multiplicity': int(pair_sites.sum()),
                }
                for drug in range(n_drug):
                    drug_rows.append(dict(window_set=name, support_subset=label,
                        drug_matrix=matrix_name, drug_index=drug,
                        drug_name=dictionary['drug_names'][drug],
                        matching_rnas=int(drug_counts[drug]),
                        drug_target_genes=int(drug_matrix.indptr[drug+1]-drug_matrix.indptr[drug])))
        summaries[name] = summary

    actions = tsv(MECHANISMS)
    edge_actions, edge_types = defaultdict(set), defaultdict(set)
    for row in actions:
        drug = int(row['drug_node_index_0based'])
        assert row['drug_name'] == dictionary['drug_names'][drug]
        gene = genes[row['gene_id']]
        assert matrices['original_mechanism'][drug, gene] == 1
        edge_actions[drug, gene].add(row['action'])
        edge_types[drug, gene].add(row['target_type'])
    assert len(edge_actions) == matrices['original_mechanism'].nnz
    action_summary = {
        'rows': len(actions), 'unique_drug_gene_edges': len(edge_actions),
        'drugs': len({d for d, g in edge_actions}), 'genes': len({g for d, g in edge_actions}),
        'actions_by_row': dict(Counter(row['action'] for row in actions).most_common()),
        'target_types_by_row': dict(Counter(row['target_type'] for row in actions).most_common()),
        'edges_with_multiple_distinct_action_strings': sum(len(v) > 1 for v in edge_actions.values()),
        'mechanism_rows_with_no_action': sum(not row['action'] for row in actions),
        'interpretation': 'Raw ChEMBL mechanism categories; no RNA effect or drug-response sign inferred. Complex-target annotations need per-gene review.'}
    matrix_summary = {name: dict(shape=list(mat.shape), edges=int(mat.nnz),
                        drugs_with_targets=int(np.count_nonzero(np.diff(mat.indptr))),
                        genes=int(len(np.unique(mat.indices)))) for name, mat in matrices.items()}
    refseq_files = Counter(path.suffix for path in (WINDOWS / 'refseq').iterdir() if path.is_file())
    audit = dict(as_of='2026-09-14', completed_at_utc=datetime.now(timezone.utc).isoformat(),
        boundary=dict(outcome_labels_read=False, predictions_read=False, evaluation_pairs_read=False,
            source_holdout_evaluated=False, new_models_fit=False,
            coverage_denominator='Full MiDrug RNA x drug dictionary Cartesian product, not an evaluation split.'),
        inputs=[dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size) for path in inputs],
        dictionary_sizes=dict(mirnas=n_rna, lncrnas=len(dictionary['rna_names']['LncDrug']), drugs=n_drug, genes=len(genes)),
        drug_matrices=matrix_summary, site_windows=summaries,
        source_filter=dict(removal, excluded_site_support_classes=dict(Counter(support_class(row['support_types']) for row in removed)),
            excluded_sites_also_nonfunctional_only=sum(support_class(row['support_types']) == 'nonfunctional_only' for row in removed)),
        drug_actions=action_summary,
        transcript_annotation=dict(cached_refseq_file_extensions=dict(refseq_files),
            cds_or_utr_annotation_audited=False,
            region_assignment_from_current_fasta_possible=False,
            limitation='FASTA + request metadata lack transcript CDS/UTR features; current sequence verification does not establish site region or original experimental isoform.'),
        cautions=['Support labels describe miRTarBase evidence categories, not RNA-drug outcome labels.',
                  'Non-Functional annotations are assay/context-specific evidence, not confirmed negative RNA-drug associations.',
                  'A shared gene is a candidate connection, not evidence of a causal drug-response mechanism.',
                  'Counts of 39 source-excluded sites and 39 nonfunctional-only sites describe different sets.'])
    csv_path = OUT / 'local_site_drug_coverage.csv'
    write_csv(csv_path, drug_rows)
    audit['outputs'] = {csv_path.name: dict(sha256=digest(csv_path), rows=len(drug_rows))}
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: audit[k] for k in ['dictionary_sizes', 'drug_matrices', 'source_filter', 'drug_actions']}, ensure_ascii=False, indent=2))
    for name, summary in summaries.items():
        print(name, json.dumps({k: summary[k] for k in ['windows', 'rna_count', 'gene_count', 'unique_rna_gene_edges', 'support_classes']}, ensure_ascii=False))
        print('coverage', json.dumps(summary['coverage'], ensure_ascii=False))


if __name__ == '__main__':
    main()
