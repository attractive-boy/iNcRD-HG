"""Support-graph matching and prediction aggregation, without model fitting."""
import numpy as np


ARMS = ('query_bag', 'fold_full', 'fold_match')
METRICS = ('auc', 'ap', 'macro_drug_ap', 'criterion')


def recipes():
    return [dict(id='frozen', arm='frozen', weight=0.)] + [
        dict(id='%s_%03d' % (arm, round(100*weight)), arm=arm, weight=weight)
        for arm in ARMS for weight in (.5, 1.)]


def support_plan(pairs, labels, fold_ids, query_pairs):
    """Return disjoint context/support indices; never accept query labels."""
    pairs, labels, fold_ids, query_pairs = map(np.asarray, (pairs, labels, fold_ids, query_pairs))
    n = len(labels)
    if pairs.shape != (2, n) or labels.shape != (n,) or fold_ids.shape != (n,):
        raise ValueError('Invalid fitting pairs, labels or folds')
    if query_pairs.ndim != 2 or query_pairs.shape[0] != 2 or not query_pairs.shape[1]:
        raise ValueError('Invalid evaluation pairs')
    if set(np.unique(labels)) != {0, 1} or set(np.unique(fold_ids)) != set(range(5)):
        raise ValueError('Expected binary labels and five complete folds')
    fitting = set(map(tuple, pairs.T))
    queries = set(map(tuple, query_pairs.T))
    if len(fitting) != n or len(queries) != query_pairs.shape[1] or fitting & queries:
        raise ValueError('Repeated or overlapping fitting/evaluation pairs')
    result = []
    for fold in range(5):
        context = np.flatnonzero(fold_ids == fold)
        support = np.flatnonzero((fold_ids != fold) & (labels == 1))
        if set(np.unique(labels[context])) != {0, 1} or not len(support):
            raise ValueError('Each context needs both classes and nonempty support')
        if set(support) & set(context):
            raise ValueError('Context row is visible to its own graph')
        result.append(dict(fold=fold, context=context, support=support))
    return result


def combine(baseline, components, recipe):
    baseline = np.asarray(baseline, dtype=np.float64)
    if baseline.ndim != 1 or not np.isfinite(baseline).all() or np.any((baseline < 0) | (baseline > 1)):
        raise ValueError('Invalid baseline probabilities')
    if recipe not in recipes():
        raise ValueError('Unregistered graph-context recipe')
    if recipe['arm'] == 'frozen':
        return baseline.copy()
    value = np.asarray(components[recipe['arm']], dtype=np.float64)
    if value.shape != baseline.shape or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError('Invalid graph-context probabilities')
    weight = recipe['weight']
    return (1-weight)*baseline + weight*value


def choose(rows, configs, seeds):
    if len(seeds) != len(set(seeds)) or len({cfg['id'] for cfg in configs}) != len(configs):
        raise ValueError('Duplicate requested seed or recipe')
    if len(rows) != len(configs)*len(seeds):
        raise ValueError('Incomplete graph-context selection grid')
    summaries = []
    for cfg in configs:
        group = [r for r in rows if r['recipe_id'] == cfg['id']]
        if len(group) != len(seeds) or sorted(r['seed'] for r in group) != sorted(seeds):
            raise ValueError('Missing or duplicate inference seed')
        value = {key: float(np.mean([r[key] for r in group]))
                 if all(r[key] is not None for r in group) else None for key in METRICS}
        if not all(value[key] is not None and np.isfinite(value[key]) for key in ('auc', 'ap', 'criterion')):
            raise ValueError('Nonfinite inner selection score')
        summaries.append(dict(recipe=cfg, **value))
    return min(summaries, key=lambda r: (-r['criterion'], r['recipe']['weight'], r['recipe']['id']))


def average_views(values, n_queries):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (5, n_queries) or not np.isfinite(values).all():
        raise ValueError('Expected five aligned query score vectors')
    if np.any((values < 0) | (values > 1)):
        raise ValueError('Invalid member probabilities')
    return values.mean(axis=0)
