"""Freeze label-free, unambiguous signed regulatory edges and binding maps."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from improve import tabular_foundation as base

ROOT = Path(__file__).resolve().parent
HERE = ROOT/'research/signed_causal_flow_20260916'
OUT = ROOT/'results/signed_regulatory_inputs_v1_20260916'
REFERENCE = ROOT/'results/tabicl_pathway_context_inputs_20260913'
ANNOTATIONS = ROOT/'results/tabicl_pathway_inputs_20260913'
DRUG = ROOT.parent.parent/'data_external/functional_priority_20260913/chembl'
NAMES = ROOT.parent.parent/'incrd-run/2.construct_network'


def unambiguous_sign(row):
    """Exclude any stimulation/inhibition conflict, even with majority consensus."""
    if not row['is_directed'] or not row['consensus_direction']:
        return 0
    positive, negative = bool(row['is_stimulation']), bool(row['is_inhibition'])
    if positive == negative:
        return 0
    if positive != bool(row['consensus_stimulation']) or negative != bool(row['consensus_inhibition']):
        return 0
    return 1 if positive else -1


def shuffled_signs(edges, signs, seed=716031):
    """A fixed receiver-stratified permutation; preserves each incoming sign count."""
    rng = np.random.default_rng(seed)
    result = signs.copy()
    for target in np.unique(edges[1]):
        positions = np.flatnonzero(edges[1] == target)
        result[positions] = rng.permutation(signs[positions])
    return result


def encode_sparse(matrix, prefix):
    matrix = matrix.tocoo()
    return {prefix+'_indices':np.stack((matrix.row,matrix.col)).astype(np.int64),
            prefix+'_values':matrix.data.astype(np.float32),
            prefix+'_shape':np.array(matrix.shape,dtype=np.int64)}


def build():
    if OUT.exists(): raise ValueError('Preserve previous input builds')
    p = base.read_json(REFERENCE/'INPUTS.json')
    assert not p['labels_read']
    for name in ('dictionary.json','MiDrug_weighted.npz','drug_gene.npz'):
        assert base.digest(REFERENCE/name) == p['files'][name]
    dp = base.read_json(DRUG/'INPUTS.json')
    assert not dp['labels_read'] and dp['reference_manifest_sha256'] == base.digest(REFERENCE/'INPUTS.json')
    assert base.digest(DRUG/'drug_gene_expanded.npz') == dp['files']['drug_gene_expanded.npz']
    dictionary = base.read_json(REFERENCE/'dictionary.json')
    assert dp['dictionary_sha256'] == base.digest(REFERENCE/'dictionary.json')
    names = [NAMES/'miRNA.csv', NAMES/'Drug.csv']
    assert pd.read_csv(names[0])['x'].tolist() == dictionary['rna_names']['MiDrug']
    assert pd.read_csv(names[1])['Drug_Name'].tolist() == dictionary['drug_names']
    genes = dictionary['genes']
    gi = {gene:i for i,gene in enumerate(genes)}
    hgnc = ANNOTATIONS/'sources/hgnc.txt.gz'
    assert base.digest(hgnc) == dp['hgnc_sha256']
    mappings = defaultdict(set)
    with gzip.open(hgnc,'rt') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if row['status'] != 'Approved' or row['locus_group'] != 'protein-coding gene': continue
            gene = row.get('ensembl_gene_id','').split('.')[0]
            if gene not in gi: continue
            for accession in row.get('uniprot_ids','').split('|'):
                if accession: mappings[accession].add(gene)
    source = HERE/'omnipath_SIGNOR_v1.tsv'
    retrieval = base.read_json(HERE/'SIGNAL_SOURCE_RETRIEVAL_V1.json')
    entry = [r for r in retrieval if Path(r.get('path','')).name == source.name][0]
    assert 'error' not in entry and base.digest(source) == entry['sha256']
    table = pd.read_csv(source,sep='\t')
    assert not table.empty and table['sources'].str.split(';').map(lambda s:'SIGNOR' in s).all()
    for column in ('is_directed','is_stimulation','is_inhibition','consensus_direction',
                   'consensus_stimulation','consensus_inhibition'):
        assert table[column].dtype == bool and not table[column].isna().any(), column
    by_pair = defaultdict(list)
    counts = Counter()
    for row in table.to_dict('records'):
        counts['downloaded_rows'] += 1
        sign = unambiguous_sign(row)
        if not sign:
            counts['direction_or_sign_ambiguous'] += 1
            continue
        mapped = [mappings.get(str(row[role]),set()) for role in ('source','target')]
        if not all(len(x) == 1 for x in mapped):
            counts['not_unique_approved_protein_mapping'] += 1
            continue
        references = str(row['references'])
        if not any(item.startswith('SIGNOR:') and item[7:].isdigit() for item in references.split(';')):
            counts['no_explicit_SIGNOR_PMID'] += 1
            continue
        s,t = [gi[next(iter(x))] for x in mapped]
        by_pair[s,t].append((sign,row))
    accepted = []
    for (s,t),items in sorted(by_pair.items()):
        signs = {x[0] for x in items}
        if len(signs) != 1:
            counts['mapped_pair_sign_conflict'] += 1
            continue
        sign = next(iter(signs))
        refs = sorted({reference for _,row in items for reference in str(row['references']).split(';')})
        accepted.append(dict(source_gene=genes[s],target_gene=genes[t],source_dictionary_index=s,target_dictionary_index=t,
            sign=sign,references=';'.join(refs),mapped_source_rows=len(items)))
    if not accepted: raise ValueError('No eligible signed regulatory interactions')
    edges = np.array([[r['source_dictionary_index'],r['target_dictionary_index']] for r in accepted],dtype=np.int64).T
    signs = np.array([r['sign'] for r in accepted],dtype=np.float32)
    rna = sparse.load_npz(REFERENCE/'MiDrug_weighted.npz').tocsr()
    drug = sparse.load_npz(DRUG/'drug_gene_expanded.npz').tocsr()
    assert rna.shape == (605,len(genes)) and drug.shape == (216,len(genes))
    assert np.isfinite(rna.data).all() and np.all(rna.data > 0)
    assert np.isfinite(drug.data).all() and np.all(drug.data > 0)
    active = np.unique(np.r_[edges.ravel(),rna.indices,drug.indices])
    relabel = np.full(len(genes),-1,dtype=np.int64)
    relabel[active] = np.arange(len(active))
    edges = relabel[edges]
    for index,row in enumerate(accepted):
        row['source_model_index'],row['target_model_index'] = map(int,edges[:,index])
    rna,drug = rna[:,active],drug[:,active]
    control = shuffled_signs(edges,signs)
    for receiver in np.unique(edges[1]):
        mask = edges[1] == receiver
        np.testing.assert_array_equal(np.sort(signs[mask]),np.sort(control[mask]))
    outgoing = np.bincount(edges[0],minlength=len(active))
    incoming = np.bincount(edges[1],minlength=len(active))
    connected = (incoming+outgoing) > 0
    coverage = dict(filters=dict(counts),genes=len(active),regulatory_edges=len(signs),
        positive_edges=int((signs > 0).sum()),negative_edges=int((signs < 0).sum()),
        curated_self_edges=int((edges[0] == edges[1]).sum()),
        rna_annotation_edges=rna.nnz,drug_binding_edges=drug.nnz,
        rna_with_binding=int((np.diff(rna.indptr) > 0).sum()),
        drug_with_binding=int((np.diff(drug.indptr) > 0).sum()),
        rna_with_regulatory_gene=int((np.asarray(rna[:,connected].sum(1)).ravel() > 0).sum()),
        drug_with_regulatory_gene=int((np.asarray(drug[:,connected].sum(1)).ravel() > 0).sum()),
        shuffled_sign_changes=int((control != signs).sum()),
        shuffled_sign_change_fraction=float((control != signs).mean()),
        receiver_stratified_sign_counts_preserved=True,
        no_binding_action_direction_inferred=True,association_labels_read=False)
    arrays = dict(regulatory_edges=edges,regulatory_sign=signs,shuffled_sign=control,
        genes=np.array([genes[i] for i in active]),rna_names=np.array(dictionary['rna_names']['MiDrug']),
        drug_names=np.array(dictionary['drug_names']),**encode_sparse(rna,'rna'),**encode_sparse(drug,'drug'))
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT/'arrays.npz',**arrays)
    with (OUT/'regulatory_edges.tsv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(accepted[0]),delimiter='\t')
        writer.writeheader();writer.writerows(accepted)
    inputs=[source,HERE/'SIGNAL_SOURCE_RETRIEVAL_V1.json',REFERENCE/'INPUTS.json',
        REFERENCE/'dictionary.json',REFERENCE/'MiDrug_weighted.npz',DRUG/'INPUTS.json',
        DRUG/'drug_gene_expanded.npz',hgnc,*names]
    base.write_json(OUT/'DATA_AUDIT.json',base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        **coverage,source='OmniPath current SIGNOR-containing interaction subset; exact SIGNOR 4.0 snapshot equivalence is not claimed',
        mapping='Unique approved HGNC protein-coding gene by full UniProt accession, no isoform stripping or symbol fallback',
        signs='Unambiguous stimulation/inhibition and matching consensus; explicit SIGNOR PMID; conflicting collapsed gene pairs omitted',
        binding_semantics='Positive annotation weights do not encode drug inhibition or RNA regulatory effect',
        original_test_source_access=False,files={name:base.digest(OUT/name) for name in ('arrays.npz','regulatory_edges.tsv')},
        source_sha256={str(Path(__file__).resolve()):base.digest(__file__)},
        input_sha256={str(x.resolve()):base.digest(x) for x in inputs})))
    print('SIGNED_REGULATORY_INPUTS',coverage,flush=True)


if __name__ == '__main__':
    build()
