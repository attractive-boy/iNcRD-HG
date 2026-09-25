"""Train-only source weighting with explicit multiple-source credit."""
from collections import Counter
import json

import numpy as np


CANDIDATES = ('source_balanced', 'source_balanced_preserve_class')
CONTROLS = ('class_prior_only', 'source_weights_shuffled')
WEIGHTINGS = ('uniform', *CANDIDATES, *CONTROLS)
REPRESENTATIONS = ('context_logistic', 'generic', 'functional_moments')


def source_lists(values):
    result = []
    for value in values:
        groups = json.loads(value) if isinstance(value, str) else list(value)
        if not isinstance(groups, list) or not groups or any(not isinstance(g, str) or not g for g in groups):
            raise ValueError('Each training triple needs a nonempty source list')
        if len(set(groups)) != len(groups):
            raise ValueError('Repeated source in a training triple')
        result.append(tuple(sorted(groups)))
    if not result:
        raise ValueError('No training triples')
    return result


def source_weights(values):
    """Give each source unit total credit before mixing multi-source triples.

    A triple citing k sources contributes 1/k to the load of each source.
    Its unnormalized sample weight is sum_s (1/k)/load_s. Thus the sum of
    unnormalized weights equals the number of distinct sources. Final weights
    have mean one so the overall regularization scale remains fixed.
    """
    groups = source_lists(values)
    loads = Counter()
    for row in groups:
        for source in row:
            loads[source] += 1.0 / len(row)
    raw = np.array([sum(1.0 / (len(row) * loads[source]) for source in row) for row in groups])
    np.testing.assert_allclose(raw.sum(), len(loads), atol=1e-9, rtol=1e-12)
    # The analytic normalizer avoids making rounding depend on row order.
    return raw * (len(groups) / len(loads)), dict(loads)


def make_weights(values, labels, seed=20260914):
    labels = np.asarray(labels)
    balanced, loads = source_weights(values)
    if labels.shape != balanced.shape or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Aligned binary training labels required')
    preserve, prior, shuffled = balanced.copy(), np.empty_like(balanced), balanced.copy()
    rng = np.random.default_rng(seed)
    for label in (0, 1):
        ids = np.flatnonzero(labels == label)
        preserve[ids] /= balanced[ids].mean()
        prior[ids] = balanced[ids].mean()
        shuffled[ids] = rng.permutation(balanced[ids])
    weights = dict(uniform=np.ones_like(balanced), source_balanced=balanced,
                   source_balanced_preserve_class=preserve, class_prior_only=prior,
                   source_weights_shuffled=shuffled)
    audit = dict(training_sources=len(loads), largest_source_fractional_load=max(loads.values()),
                 source_loads=loads, schemes={})
    for name, weight in weights.items():
        np.testing.assert_allclose(weight.sum(), len(labels), atol=1e-9, rtol=1e-12)
        audit['schemes'][name] = dict(min_weight=float(weight.min()), max_weight=float(weight.max()),
            effective_sample_size=float(weight.sum()**2 / np.square(weight).sum()),
            class_mass={str(label): float(weight[labels == label].sum()) for label in (0, 1)},
            weighted_resistant_fraction=float(np.average(labels, weights=weight)))
    return weights, audit


def partition_ids(frame, column):
    return {value for cell in frame[column] for value in json.loads(cell)}


def audit_partitions(train, validation):
    result = {}
    for column in ('source_groups', 'record_uids', 'pmids'):
        overlap = partition_ids(train, column) & partition_ids(validation, column)
        if overlap:
            raise ValueError('Training/validation source overlap: ' + column)
        result[column + '_intersection'] = 0
    if validation.novel_triple.dtype != bool:
        raise ValueError('Novel-triple eligibility must be an explicit boolean column')
    primary_ids = set(validation.loc[validation.novel_triple, 'triple_id'])
    if primary_ids & set(train.triple_id):
        raise ValueError('Primary validation contains a training triple')
    result['primary_train_triple_intersection'] = 0
    return result
