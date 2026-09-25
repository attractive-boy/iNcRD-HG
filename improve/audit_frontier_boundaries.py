"""Audit actual saved projection support sets and frozen validation artifacts."""
from pathlib import Path

import numpy as np
import pandas as pd

from improve import tabular_foundation as base
from improve import tabicl_context_program as accepted
from improve import tabicl_pathway as operations
from improve import frozen_feature_validation as experiment
from improve import site_evidence_features as sites


RESULTS = Path(__file__).resolve().parent/'results'


def run():
    out = RESULTS/'frontier_validation_20260914'
    out.mkdir(exist_ok=True)
    target = out/'BOUNDARY_AUDIT.json'
    if target.exists():
        raise ValueError('Boundary audit already recorded')
    audit,files,counts = {},{},{}
    for family,name,expected in [('site','site_evidence_association_20260914',144),
                                 ('drug','drug_multimodal_association_20260914',105)]:
        directory = RESULTS/name
        p = experiment.protocol(directory)
        validation = base.read_json(directory/'VALIDATION.json')
        if validation['protocol_sha256']!=p['sha256'] or not validation['passed']:
            raise ValueError('Invalid completed association evaluation')
        actual = sorted((directory/'inner').glob('**/result.json'))
        if len(actual)!=expected or validation['fitted_states']!=expected:
            raise ValueError('Formal job count mismatch')
        rows = []
        for task in p['tasks']:
            a,_ = accepted.stage_data(directory,p,task,'inner')
            enc = experiment.encoder(p,task)
            for variant in p['variants']:
                for kind in operations.kinds(p,task):
                    keep = np.arange(len(a['y_train'])) if kind=='full' else np.flatnonzero(a['fold_ids']==int(kind[-1]))
                    pairs = a['pairs_train'][:,keep]
                    if family=='site':
                        table,_ = enc.table(variant,pairs[0])
                        keys = pairs[0]*enc.n_drugs+pairs[1] if variant.startswith('target') else pairs[0]
                        fitted = np.intersect1d(keys,table['ids'])
                    else:
                        keys = pairs[1]
                        fitted = np.unique(keys[enc.present[keys]]) if variant!='descriptors' else np.array([],int)
                    for seed in p['seeds']:
                        dest = operations.directory(directory,'inner',task,variant,seed,kind)
                        result = operations.checked_result(dest,p)
                        if result['parameter_sha256']!=p['parameter_sha256'] or result['input_columns']!=86:
                            raise ValueError('Model budget/parameter mismatch')
                        with np.load(dest/'projection.npz',allow_pickle=False) as z:
                            saved = z['fit_entity_ids'] if family=='site' else (z['fit_ids'] if 'fit_ids' in z.files else np.array([],int))
                            np.testing.assert_array_equal(saved,fitted)
                            if family=='drug' and variant=='m2umol_shuffled':
                                membership = np.zeros(enc.n_drugs,bool)
                                membership[np.unique(keys)] = True
                                np.testing.assert_array_equal(membership[z['permutation']],membership)
                            if family=='site' and variant=='evidence_shuffled':
                                support = np.zeros(enc.n_rnas,bool)
                                support[np.unique(pairs[0])] = True
                                rna = enc.sites['rna']
                                np.testing.assert_array_equal(support[rna[z['permutation']]],support[rna])
                        rows.append(dict(task=task,variant=variant,kind=kind,seed=seed,
                            support_rows=len(keep),unique_support_entities=len(np.unique(keys)),
                            fitted_covered_entities=len(fitted),replay_max_abs=result['replay_max_abs']))
        for name_,digest in validation['artifacts'].items():
            if base.digest(directory/name_)!=digest:
                raise ValueError('Completed association artifact changed: '+name_)
        audit[family] = rows
        counts[family] = len(rows)
        files[str(directory/'VALIDATION.json')] = base.digest(directory/'VALIDATION.json')
    directory = RESULTS/'site_evidence_source_20260914'
    p = base.check_seal(base.read_json(directory/'protocol.json'))
    validation = base.read_json(directory/'EVALUATION.json')
    for name,digest in validation['artifacts'].items():
        if base.digest(directory/name)!=digest:
            raise ValueError('Completed source artifact changed')
    enc = sites.SiteFeatures(p['representation_dir'],'source')
    train = pd.read_csv(Path(p['source_dir'])/'MiDir/train.tsv',sep='\t',usecols=['ncRNA','cid'],dtype=str)
    rna_map = {n:i for i,n in enumerate(enc.sites['names'])}
    drug_map = {cid:i for i,cid in enumerate(enc.sites['cids'])}
    pairs = np.stack([train.ncRNA.map(rna_map).to_numpy(int),train.cid.map(drug_map).to_numpy(int)])
    audit['source'] = []
    for variant in p['variants']:
        table,_ = enc.table(variant,pairs[0])
        keys = pairs[0]*enc.n_drugs+pairs[1] if variant.startswith('target') else pairs[0]
        expected = np.intersect1d(keys,table['ids'])
        with np.load(directory/'MiDir'/variant/'projection.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['fit_entity_ids'],expected)
        audit['source'].append(dict(variant=variant,fitted_covered_entities=len(expected)))
    counts['source'] = len(audit['source'])
    files[str(directory/'EVALUATION.json')] = base.digest(directory/'EVALUATION.json')
    smoke = {}
    for name in ('site_evidence_smoke_20260914','drug_multimodal_smoke_20260914'):
        path = RESULTS/name/'SMOKE_VALIDATION.json'
        v = base.read_json(path)
        if not v['passed']:
            raise ValueError('Smoke checks failed')
        for name_,digest in v['artifacts'].items():
            if base.digest(path.parent/name_)!=digest:
                raise ValueError('Smoke artifact changed')
        smoke[name] = {key:v[key] for key in ('fitted_states','max_replay','max_query_invariance')}
        files[str(path)] = base.digest(path)
    base.write_json(target,dict(passed=True,source_sha256=base.digest(__file__),checked_contexts=counts,
        all_saved_fit_sets_equal_unique_covered_support=True,query_only_projection_fit=False,
        all_tabicl_parameters_match_reference=True,smoke=smoke,details=audit,files=files,
        protected_scope='Association loaders restricted to inner; source projection audit reads train identity columns only. No historical outer or original association test arrays, or reserved source holdout outcomes.',
        accepted_reference_replaced=False))
    print('BOUNDARIES_VERIFIED',counts,flush=True)


if __name__=='__main__':
    run()
