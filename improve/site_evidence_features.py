"""Evidence- and target-aware site representations with support-only projection.

This module never loads association outcomes or any reserved evaluation partition.
Site evidence classes describe the auxiliary assay, not RNA--drug response labels.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy import sparse

from improve import tabular_foundation as base
from improve import rna_functional_inputs as original_inputs
from improve import rna_functional_modules as original_features
from improve.context_functional_eval import source_mask


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / 'results/site_evidence_inputs_20260914'
REPRESENTATIONS = ROOT / 'results/rna_functional_inputs_20260914_v2'
CONTEXT_INPUTS = ROOT / 'results/tabicl_pathway_context_inputs_20260913'
FILTERED = ROOT / 'results/context_functional_20260914'
EXPANDED = ROOT.parents[1] / 'data_external/functional_priority_20260913/chembl'
DRUGS = ROOT.parents[1] / 'incrd-run/2.construct_network/Drug.csv'
VARIANTS = ('generic', 'flat', 'evidence', 'evidence_stats', 'evidence_shuffled',
            'target', 'target_stats', 'target_shuffled')
EVIDENCE = ('strong_functional', 'weak_functional', 'nonfunctional_only', 'mixed')
WIDTH = 30
CHANNEL_WIDTH = WIDTH * 2 + 3
PCS = 14
SEED = 20260914


def evidence_class(value):
    tokens = {x.strip() for x in str(value).split(';') if x.strip()}
    allowed = {'Functional MTI', 'Functional MTI (Weak)',
               'Non-Functional MTI', 'Non-Functional MTI (Weak)'}
    if not tokens or not tokens <= allowed:
        raise ValueError('Unknown site evidence tokens: ' + repr(tokens))
    positive = bool(tokens & {'Functional MTI', 'Functional MTI (Weak)'})
    negative = bool(tokens & {'Non-Functional MTI', 'Non-Functional MTI (Weak)'})
    if positive and negative:
        return 3
    if negative:
        return 2
    return 0 if 'Functional MTI' in tokens else 1


def permute_groups(groups, identities, seed=SEED):
    """Maximal identity mismatches by rotating sorted identity blocks.

    Only rows in the same group exchange vectors. Dominant identities and
    single-identity groups can leave unavoidable matches; report them explicitly.
    """
    identities = np.asarray(identities)
    lookup = {}
    for i, group in enumerate(groups):
        lookup.setdefault(tuple(np.atleast_1d(group).tolist()), []).append(i)
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(identities))
    eligible = 0
    for key in sorted(lookup):
        ids = np.asarray(lookup[key], int)
        labels = np.unique(identities[ids])
        if len(labels) < 2:
            continue
        eligible += len(ids)
        blocks = [rng.permutation(ids[identities[ids] == label])
                  for label in rng.permutation(labels)]
        order = np.concatenate(blocks)
        permutation[order] = np.roll(order, max(map(len, blocks)))
    changed = identities[permutation] != identities
    return permutation, dict(seed=seed,rows=len(identities),groups=len(lookup),
        eligible_rows=eligible,identity_changed=int(changed.sum()),
        identity_changed_fraction=float(changed.mean()) if len(changed) else 0.,
        vector_rows_moved=int(np.sum(permutation != np.arange(len(permutation)))),
        null='One deterministic constrained permutation; not an estimated null distribution.')


def pool(vectors, entities, channels, genes, n_channels):
    """Aggregate selected site rows; keep a sparse table of covered entities."""
    vectors = np.asarray(vectors, np.float64)
    entities, channels, genes = (np.asarray(v) for v in (entities, channels, genes))
    if not (len(vectors) == len(entities) == len(channels) == len(genes)):
        raise ValueError('Site correspondence mismatch')
    if vectors.ndim != 2 or vectors.shape[1] != WIDTH:
        raise ValueError('Expected 30-dimensional cached site states')
    ids, inverse = np.unique(entities, return_inverse=True)
    bins = inverse * n_channels + channels
    size = len(ids) * n_channels
    counts = np.bincount(bins, minlength=size)
    sums, squared = np.zeros((size, WIDTH)), np.zeros((size, WIDTH))
    np.add.at(sums, bins, vectors)
    np.add.at(squared, bins, vectors * vectors)
    denom = np.maximum(counts, 1)[:, None]
    mean = sums / denom
    sd = np.sqrt(np.maximum(squared / denom - mean * mean, 0))
    gene_counts = np.zeros(size, int)
    total_gene_counts = np.zeros(len(ids), int)
    if len(entities):
        # Integer gene codes avoid lexicographic conversion of numeric bin IDs.
        _, gene_codes = np.unique(genes, return_inverse=True)
        unique = np.unique(np.stack([bins, gene_codes], axis=1), axis=0)
        gene_counts = np.bincount(unique[:, 0], minlength=size)
        unique_total = np.unique(np.stack([inverse, gene_codes], axis=1), axis=0)
        total_gene_counts = np.bincount(unique_total[:, 0], minlength=len(ids))
    payload = np.column_stack([mean, sd, counts > 0, np.log1p(counts),
                               np.log1p(gene_counts)])
    return dict(ids=ids.astype(int),payload=payload.reshape(len(ids), n_channels * CHANNEL_WIDTH).astype(np.float32),
                gene_counts=total_gene_counts.astype(int),
                counts=counts.reshape(len(ids), n_channels).astype(int))


def target_links(rna_ids, gene_names, categories, dictionary, matrices):
    gene_map = {g:i for i,g in enumerate(dictionary['genes'])}
    n_drugs = len(dictionary['drug_names'])
    site_ids, entities, channels = [], [], []
    for tier, matrix in enumerate(matrices):
        matrix = matrix.tocsc()
        for i, (rna, gene, category) in enumerate(zip(rna_ids, gene_names, categories)):
            column = gene_map.get(gene)
            if column is None:
                continue
            drugs = matrix.indices[matrix.indptr[column]:matrix.indptr[column + 1]]
            site_ids.extend([i] * len(drugs))
            entities.extend((rna * n_drugs + drugs).tolist())
            channels.extend([tier * 4 + category] * len(drugs))
    return tuple(np.asarray(v, int) for v in (site_ids, entities, channels))


def write_archive(path, table):
    np.savez_compressed(path, **table)
    return base.digest(path)


def prepare(out=DEFAULT_OUT):
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new site input directory')
    original_inputs.verify(REPRESENTATIONS / 'windows')
    original_inputs.verify(REPRESENTATIONS / 'mirbind')
    original_inputs.verify(REPRESENTATIONS / 'assembled')
    dictionary = base.read_json(CONTEXT_INPUTS / 'dictionary.json')
    expansion = base.read_json(EXPANDED / 'INPUTS.json')
    if expansion['dictionary_sha256'] != base.digest(CONTEXT_INPUTS / 'dictionary.json'):
        raise ValueError('Expanded target dictionary mismatch')
    original = sparse.load_npz(CONTEXT_INPUTS / 'drug_gene.npz').astype(bool)
    expanded = sparse.load_npz(EXPANDED / 'drug_gene_expanded.npz').astype(bool)
    if original.shape != expanded.shape or original.shape != (len(dictionary['drug_names']),len(dictionary['genes'])):
        raise ValueError('Target matrix dimensions do not match the frozen dictionary')
    if original.multiply(expanded).nnz != original.nnz:
        raise ValueError('Expansion removed original target edges')
    added = expanded.astype(np.int8) - original.astype(np.int8)
    added.eliminate_zeros()
    rows = original_inputs.site_rows(REPRESENTATIONS)
    with np.load(REPRESENTATIONS / 'mirbind/representations.npz', allow_pickle=False) as z:
        vectors = z['features']
        np.testing.assert_array_equal(z['site_ids'], [r['site_id'] for r in rows])
    filter_record = base.read_json(FILTERED / 'SOURCE_FILTER.json')
    keep_source, source_counts = source_mask(rows, set(filter_record['forbidden_pmids']))
    filtered_file = FILTERED / 'representations_MiDir.npz'
    fp = base.read_json(FILTERED / 'protocol.json')
    if base.digest(filtered_file) != fp['files']['representations_MiDir.npz']:
        raise ValueError('Source-filtered RNA table changed')
    if base.digest(FILTERED / 'SOURCE_FILTER.json') != fp['files']['SOURCE_FILTER.json']:
        raise ValueError('Frozen source exclusion metadata changed')
    drugs = pd.read_csv(DRUGS, dtype=str).set_index('Drug_Name')
    cids = drugs.loc[dictionary['drug_names'],'CID'].to_numpy(str)
    if len(set(cids)) != len(cids):
        raise ValueError('Ambiguous drug CID mapping')
    out.mkdir(parents=True)
    files, audits = {}, {}
    for mode, keep in [('association', np.ones(len(rows), bool)), ('source', keep_source)]:
        directory = out / mode
        directory.mkdir()
        selected = [r for i, r in enumerate(rows) if keep[i]]
        x = vectors[keep]
        rna = np.asarray([int(r['rna_index']) for r in selected])
        genes = np.asarray([r['gene_id'] for r in selected])
        categories = np.asarray([evidence_class(r['support_types']) for r in selected])
        source_table = REPRESENTATIONS / 'assembled/MiDrug.npz' if mode == 'association' else filtered_file
        with np.load(source_table, allow_pickle=False) as z:
            table = {k:z[k] for k in z.files}
        np.testing.assert_array_equal(table['names'], dictionary['rna_names']['MiDrug'])
        totals = pool(x, rna, np.zeros(len(x), int), genes, 1)
        np.testing.assert_array_equal(totals['ids'],np.flatnonzero(table['present']))
        np.testing.assert_allclose(totals['payload'][:,:WIDTH],table['functional_mean'][totals['ids']],atol=2e-6)
        meta = dict(simple=table['simple'], names=table['names'], cids=cids,
                    vectors=x, rna=rna, genes=genes, categories=categories,
                    site_ids=np.asarray([r['site_id'] for r in selected]))
        files[mode + '/sites.npz'] = write_archive(directory / 'sites.npz',meta)
        for name, payload in [('generic',table['generic']),('flat',table['functional_moments'])]:
            pack = dict(ids=totals['ids'],payload=payload[totals['ids']],gene_counts=totals['gene_counts'],counts=totals['counts'])
            files[mode + '/' + name + '.npz'] = write_archive(directory / (name + '.npz'),pack)
        evidence = pool(x,rna,categories,genes,4)
        files[mode + '/evidence.npz'] = write_archive(directory / 'evidence.npz',evidence)
        sid, pairs, channel = target_links(rna,genes,categories,dictionary,[original,added])
        target = pool(x[sid],pairs,channel,genes[sid],8)
        files[mode + '/target.npz'] = write_archive(directory / 'target.npz',target)
        permutation, shuffle = permute_groups(np.column_stack([rna,categories]),genes)
        # Matching vector-to-gene assignments changes, not the original topology.
        shuffled = pool(x[permutation][sid],pairs,channel,genes[sid],8)
        np.testing.assert_array_equal(shuffled['counts'],target['counts'])
        np.testing.assert_array_equal(shuffled['gene_counts'],target['gene_counts'])
        check = pool(x[permutation],rna,categories,genes,4)
        np.testing.assert_allclose(check['payload'],evidence['payload'],atol=2e-6)
        shuffled['permutation'] = permutation
        files[mode + '/target_shuffled.npz'] = write_archive(directory / 'target_shuffled.npz',shuffled)
        changed = np.any(np.abs(shuffled['payload'] - target['payload']) > 1e-6,axis=1)
        audits[mode] = dict(sites=len(selected),evidence_counts={name:int(np.sum(categories==i)) for i,name in enumerate(EVIDENCE)},
            rnas=len(totals['ids']),genes=len(set(genes)),unknown_dictionary_genes=sorted(set(genes)-set(dictionary['genes'])),
            target_pairs=len(target['ids']),total_pairs=len(table['names'])*len(cids),
            target_pair_fraction=len(target['ids'])/(len(table['names'])*len(cids)),
            target_vector_changed_pairs=int(changed.sum()),target_shuffle=shuffle)
    inputs = [CONTEXT_INPUTS/'dictionary.json',CONTEXT_INPUTS/'drug_gene.npz',EXPANDED/'drug_gene_expanded.npz',
        EXPANDED/'INPUTS.json',REPRESENTATIONS/'windows/VALIDATION.json',REPRESENTATIONS/'mirbind/VALIDATION.json',
        REPRESENTATIONS/'assembled/VALIDATION.json',FILTERED/'SOURCE_FILTER.json',filtered_file,DRUGS]
    record = base.seal(dict(files=files,inputs={str(p):base.digest(p) for p in inputs},audit=audits,
        source_filter=source_counts,source_code_sha256=base.digest(__file__),source_labels_loaded=False,
        outcome_labels_loaded=False,original_target_edges=original.nnz,added_binding_edges=added.nnz,
        evidence_groups=list(EVIDENCE),channel_payload='30 mean + 30 SD + presence + log1p(site count) + log1p(unique gene count)',
        dimensions='24 output columns: original 8 RNA covariates + explicit pooled coverage and log1p(gene count) + 14 support-only PCs.',
        unknown_action='No causal signs inferred from binding, complex/family mechanisms or nonfunctional site categories.'))
    base.write_json(out/'INPUTS.json',record)
    print(json.dumps(audits,indent=2),flush=True)


def verify(directory):
    directory = Path(directory)
    p = base.check_seal(base.read_json(directory / 'INPUTS.json'))
    for name, digest in p['files'].items():
        if base.digest(directory / name) != digest:
            raise ValueError('Site input changed: ' + name)
    for name, digest in p['inputs'].items():
        if base.digest(name) != digest:
            raise ValueError('Upstream site input changed: ' + name)
    return p


class SiteFeatures:
    def __init__(self,directory,mode):
        self.directory = Path(directory) / mode
        with np.load(self.directory/'sites.npz',allow_pickle=False) as z:
            self.sites = {k:z[k] for k in z.files}
        self.n_rnas, self.n_drugs = len(self.sites['names']), len(self.sites['cids'])
        self.cache = {}

    def table(self,variant,support_rnas):
        if variant == 'evidence_shuffled':
            s = self.sites
            support = np.zeros(self.n_rnas,int)
            support[np.unique(support_rnas)] = 1
            groups = np.column_stack([s['categories'],support[s['rna']]])
            permutation, audit = permute_groups(groups,s['rna'])
            data = pool(s['vectors'][permutation],s['rna'],s['categories'],s['genes'],4)
            data['permutation'] = permutation
            return data,audit
        key = variant.replace('_stats','')
        if key not in self.cache:
            with np.load(self.directory/(key+'.npz'),allow_pickle=False) as z:
                self.cache[key] = {k:z[k] for k in z.files}
        data = self.cache[key]
        if variant.endswith('_stats'):
            data = dict(data)
            payload = data['payload'].copy().reshape(len(data['ids']),-1,CHANNEL_WIDTH)
            payload[:,:,:2*WIDTH] = 0
            data['payload'] = payload.reshape(len(data['ids']),-1)
        return data,{}

    def transform(self,variant,support_pairs):
        if variant not in VARIANTS:
            raise ValueError('Unregistered site representation')
        support_pairs = np.asarray(support_pairs,int)
        if support_pairs.ndim != 2 or support_pairs.shape[0] != 2:
            raise ValueError('Expected support RNA/drug pairs')
        pairwise = variant.startswith('target')
        size = self.n_rnas*self.n_drugs if pairwise else self.n_rnas
        support = support_pairs[0]*self.n_drugs+support_pairs[1] if pairwise else support_pairs[0]
        data, shuffle = self.table(variant,support_pairs[0])
        ids = data['ids']
        index = np.full(size,-1,int)
        index[ids] = np.arange(len(ids))
        fit = index[np.unique(support)]
        fit = fit[fit>=0]
        # Guard the constant-payload case without manufacturing NaN variance.
        if len(fit)<2 or not np.any(np.var(data['payload'][fit],axis=0)>0):
            projected = np.zeros((len(ids),PCS),np.float32)
            state,meta = dict(fit_ids=fit),dict(fit_entities=len(fit),components=0,explained_variance_fraction=0.)
        else:
            projected,state,meta = original_features.project(data['payload'],np.ones(len(ids),bool),fit,PCS)
        extra = np.zeros((size,2+PCS),np.float32)
        extra[ids,0] = 1
        extra[ids,1] = np.log1p(data['gene_counts'])
        extra[ids,2:] = projected
        simple = np.repeat(self.sites['simple'],self.n_drugs,axis=0) if pairwise else self.sites['simple']
        state.update(fit_entity_ids=ids[fit],entity_ids=ids)
        if 'permutation' in data:
            state['permutation'] = data['permutation']
        meta.update(shuffle=shuffle,pairwise=pairwise,explicit_columns=['pooled_present','log1p_unique_pooled_genes'],
                    support_ids_sha256=base.json_digest(np.unique(support).tolist()),
                    projection_columns=PCS,output_columns=24)
        return np.concatenate([simple,extra],axis=1),state,meta

    def select(self,table,pairs,variant):
        ids = pairs[0]*self.n_drugs+pairs[1] if variant.startswith('target') else pairs[0]
        return table[ids]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',default=str(DEFAULT_OUT))
    prepare(parser.parse_args().out_dir)
