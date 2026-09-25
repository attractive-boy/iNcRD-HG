"""A shared evidence-tiered pathway adapter with expression-context features."""
from pathlib import Path

import numpy as np
from scipy import sparse

from improve import tabular_foundation as base
from improve.pathway_context_data import CONTEXT_NAMES, TIERS, WEIGHTS
from improve.tabicl_pathway_modules import ProgramPairs, DIRECT_NAMES, binary, assess as original_assess


VARIANTS = ('annotation', 'tiered', 'context', 'context_permuted')
CANDIDATES = ('annotation', 'tiered', 'context')
STRENGTHS = (.001, .005, .01, .025, .05, .1, .25, .5, 1.)
TIER_FEATURES = ('rna_present', 'rna_log_target_count', 'drug_gene_cosine', 'drug_program_cosine')
PERMUTATION_SEED = 20260913


def names(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown evidence adapter variant')
    result = list(DIRECT_NAMES)
    if variant != 'annotation':
        result += [tier+'_'+feature for tier in TIERS for feature in TIER_FEATURES]
    if variant.startswith('context'):
        result += list(CONTEXT_NAMES)
    return result


def recipes():
    return [dict(id='baseline', variant='baseline', alpha=0.)] + [
        dict(id=variant+'_'+format(alpha, '.4f').replace('.', 'p'), variant=variant, alpha=alpha,
             role='candidate' if variant in CANDIDATES else 'negative_control')
        for variant in VARIANTS for alpha in STRENGTHS]


class ContextPrograms:
    """Both RNA types use identical operators, weights, and feature semantics."""
    def __init__(self, gp, dg, tiers, context):
        self.gp, self.dg = binary(gp), binary(dg)
        self.tiers, self.context = tiers, context
        if set(tiers) != set(context):
            raise ValueError('Expression/evidence task mismatch')
        self.tables = {}
        for task, layers in tiers.items():
            if set(layers) != set(TIERS):
                raise ValueError('All evidence tiers must be explicitly present')
            shape = layers['functional'].shape
            if shape[1] != self.gp.shape[0] or any(x.shape != shape for x in layers.values()):
                raise ValueError('Gene dictionaries do not match')
            if context[task].shape != (shape[0], self.dg.shape[0], len(CONTEXT_NAMES)):
                raise ValueError('Expression feature dimensions do not match')
            if not np.isfinite(context[task]).all():
                raise ValueError('Nonfinite context features')
            if any(not np.isfinite(x.data).all() or (x.data < 0).any() for x in layers.values()):
                raise ValueError('Invalid annotation evidence')

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = base.read_json(directory/'INPUTS.json')
        for name, expected in manifest['files'].items():
            if base.digest(directory/name) != expected:
                raise ValueError('Frozen context evidence changed: '+name)
        dictionary = base.read_json(directory/'dictionary.json')
        load = lambda name: sparse.load_npz(directory/(name+'.npz'))
        result = cls(load('gene_pathway'), load('drug_gene'),
                     {task:{tier:load(task+'_'+tier) for tier in TIERS} for task in dictionary['rna_names']},
                     {task:np.load(directory/(task+'_expression_features.npy'), allow_pickle=False)
                      for task in dictionary['rna_names']})
        if len(dictionary['genes']) != result.gp.shape[0] or len(dictionary['drug_names']) != result.dg.shape[0]:
            raise ValueError('Entity dictionary mismatch')
        for task in result.tiers:
            if len(dictionary['rna_names'][task]) != result.context[task].shape[0]:
                raise ValueError('RNA dictionary mismatch')
        return result

    def table(self, task, variant):
        if variant not in VARIANTS or task not in self.tiers:
            raise ValueError('Unknown task or variant')
        key = task, variant
        if key in self.tables:
            return self.tables[key]
        layers = self.tiers[task]
        union = binary(sum(layers.values()))
        weighted = sparse.csr_matrix(union.shape, dtype=np.float64)
        for tier in TIERS:
            weighted = weighted.maximum(layers[tier]*WEIGHTS[tier])
        zero = sparse.csr_matrix(union.shape)
        kernel = ProgramPairs(self.gp, self.dg, {task:union}, {task:zero})
        annotation, _ = kernel.group(union)
        self.tables[task,'annotation'] = annotation
        if variant == 'annotation':
            return annotation
        main, _ = kernel.group(weighted)
        extra = [kernel.group(layers[tier])[0][:,:,[0,2,5,11]] for tier in TIERS]
        tiered = np.concatenate([main,*extra], axis=-1).astype(np.float32)
        self.tables[task,'tiered'] = tiered
        if variant == 'tiered':
            return tiered
        expression = self.context[task]
        if variant == 'context_permuted':
            permutation = np.random.default_rng(PERMUTATION_SEED).permutation(len(expression))
            expression = expression[permutation]
        result = np.concatenate([tiered, expression], axis=-1).astype(np.float32)
        if result.shape[-1] != len(names(variant)):
            raise ValueError('Feature name/shape mismatch')
        self.tables[key] = result
        return result

    def transform(self, task, pairs, variant):
        pairs = np.asarray(pairs)
        if pairs.ndim != 2 or pairs.shape[0] != 2 or not np.issubdtype(pairs.dtype, np.integer):
            raise ValueError('Expected integer RNA/drug indices with shape (2,n)')
        table = self.table(task, variant)
        if (pairs < 0).any() or (pairs[0] >= table.shape[0]).any() or (pairs[1] >= table.shape[1]).any():
            raise ValueError('Entity index outside frozen dictionary')
        return table[pairs[0], pairs[1]].copy()


def assess(rows, config, seeds):
    result = original_assess(rows, config, seeds)
    result['eligible'] = (result['eligible'] and config['variant'] in CANDIDATES and
        all(result['metrics']['LncDrug'][m]['nonnegative_seeds'] == len(seeds) for m in ('auc','ap')))
    result['lnc_per_seed_required'] = True
    result['role'] = config.get('role', 'candidate')
    return result
