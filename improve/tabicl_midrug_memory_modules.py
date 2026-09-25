"""Drug-routed context memory and a protected residual around frozen TabICLv2.

This module never accepts evaluation labels. The stable LncDrug path is an
identity, not a validation-dependent rollback. All routing is per drug and
therefore independent of the other queries in an inference batch.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


COMPONENTS = ('drug_graph', 'drug_side')


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected a finite matrix')
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def probabilities(values):
    x = np.asarray(values)
    if x.ndim != 1 or not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError('Expected finite one-dimensional probabilities')
    return x


def logit(values):
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-7, 1-1e-7)
    return np.log(values) - np.log1p(-values)


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    return np.exp(-np.logaddexp(0., -values))


def check_pairs(pairs, nr, nd):
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[0] != 2 or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError('Expected integer RNA/drug pairs')
    if np.any(pairs < 0) or np.any(pairs[0] >= nr) or np.any(pairs[1] >= nd):
        raise ValueError('Pair index outside entity inputs')
    return pairs


def fit_memory(drug_features, rna_features, fitting_pairs, *, clusters=8,
               context_size=2048, reservoir_size=256, side_dim=8, seed=20260912):
    """Fit label-free drug routing and retrieve contexts from fitting rows only.

    Entity attributes are the same fixed, transductively available attributes
    as the reference. No target adjacency, fitting labels, or query pairs are
    needed for routing. The small global reservoir prevents overly narrow
    local contexts without using class-balanced sampling.
    """
    drug, rna = np.asarray(drug_features), np.asarray(rna_features)
    if drug.ndim != 2 or rna.ndim != 2 or not np.isfinite(drug).all() or not np.isfinite(rna).all():
        raise ValueError('Invalid entity features')
    pairs = check_pairs(fitting_pairs, len(rna), len(drug))
    if len(set(map(tuple, pairs.T))) != pairs.shape[1]:
        raise ValueError('Duplicate fitting pairs')
    if not 1 <= clusters <= len(drug) or not 0 <= reservoir_size < context_size or context_size < 2:
        raise ValueError('Invalid memory sizes')
    if not 1 <= side_dim <= min(drug.shape[0], drug.shape[1], rna.shape[1]):
        raise ValueError('Invalid side embedding dimension')
    du = unit(drug)
    kmeans = KMeans(n_clusters=clusters, n_init=10, random_state=seed).fit(du)
    centers = unit(kmeans.cluster_centers_)
    similarities = du @ centers.T
    rng = np.random.default_rng(seed)
    n = pairs.shape[1]
    reservoir = np.sort(rng.choice(n, min(reservoir_size, n), replace=False))
    available = np.setdiff1d(np.arange(n), reservoir)
    contexts = []
    for group in range(clusters):
        order = np.argsort(-similarities[pairs[1, available], group], kind='stable')
        local = available[order[:max(0, min(context_size, n)-len(reservoir))]]
        contexts.append(np.sort(np.r_[reservoir, local]).astype(np.int64))
    pca = PCA(n_components=side_dim, svd_solver='full').fit(drug)
    return dict(centers=centers, similarities=similarities,
                contexts=np.stack(contexts), reservoir=reservoir,
                drug_embedding=pca.transform(drug).astype(np.float32),
                rna_embedding=rna[:, :side_dim].astype(np.float32).copy(),
                drug_pca_mean=pca.mean_, drug_pca_components=pca.components_)


def route(memory, pairs, *, topk=2, temperature=.1):
    pairs = check_pairs(pairs, len(memory['rna_embedding']), len(memory['drug_embedding']))
    similarities = np.asarray(memory['similarities'])[pairs[1]]
    if not 1 <= topk <= similarities.shape[1] or temperature <= 0:
        raise ValueError('Invalid routing parameters')
    chosen = np.argsort(-similarities, axis=1, kind='stable')[:, :topk]
    logits = np.take_along_axis(similarities, chosen, axis=1) / temperature
    weights = np.exp(logits-logits.max(1, keepdims=True))
    weights /= weights.sum(1, keepdims=True)
    dense = np.zeros_like(similarities)
    np.put_along_axis(dense, chosen, weights, axis=1)
    reliability = np.clip(np.max(similarities, axis=1), 0., 1.)
    return dense, reliability


def append_side(x, pairs, memory):
    x = np.asarray(x)
    pairs = check_pairs(pairs, len(memory['rna_embedding']), len(memory['drug_embedding']))
    if x.ndim != 2 or len(x) != pairs.shape[1] or not np.isfinite(x).all():
        raise ValueError('Invalid pair feature matrix')
    return np.concatenate([x, memory['rna_embedding'][pairs[0]],
                           memory['drug_embedding'][pairs[1]]], axis=1).astype(np.float32)


def recipes():
    return [dict(id='frozen', component='frozen', alpha=0., gate='identity')] + [
        dict(id='%s_%s_%03d' % (component, gate, round(100*alpha)),
             component=component, alpha=alpha, gate=gate)
        for component in COMPONENTS for gate in ('consensus', 'uniform')
        for alpha in (.05, .1, .2, .35)]


def protected_residual(task, baseline, views, reliability, recipe):
    """Preserve LncDrug exactly; make a bounded correction for MiDrug.

    The local probability is averaged over five support graphs. Disagreement
    between their logits reduces the consensus gate. No learned parameters or
    evaluation labels enter this gate. A zero-strength path returns the input
    scores directly, avoiding even sigmoid/logit round-off.
    """
    baseline = probabilities(baseline)
    if task not in ('LncDrug', 'MiDrug'):
        raise ValueError('Unknown target relation')
    if task == 'LncDrug':
        return baseline.copy()
    if recipe not in recipes():
        raise ValueError('Unregistered residual recipe')
    if recipe['alpha'] == 0:
        return baseline.copy()
    views = np.asarray(views, dtype=np.float64)
    reliability = np.asarray(reliability, dtype=np.float64)
    if (views.shape != (5, len(baseline)) or reliability.shape != baseline.shape
            or not np.isfinite(views).all() or not np.isfinite(reliability).all()
            or np.any((views < 0) | (views > 1)) or np.any((reliability < 0) | (reliability > 1))):
        raise ValueError('Invalid expert probabilities or reliability')
    gate = np.ones(len(baseline))
    if recipe['gate'] == 'consensus':
        gate = reliability / (1.+np.std(logit(views), axis=0))
    delta = np.clip(logit(views.mean(0))-logit(baseline), -2., 2.)
    result = sigmoid(logit(baseline)+recipe['alpha']*gate*delta)
    # Preserve an unavailable expert exactly, including endpoint probabilities.
    return np.where(gate == 0., baseline, result)


def choose(rows, candidates, seeds):
    """Require both inner AUC and AP gains; otherwise retain the stable path."""
    if len(seeds) != len(set(seeds)) or len({c['id'] for c in candidates}) != len(candidates):
        raise ValueError('Duplicate seed or recipe')
    if len(rows) != len(seeds)*len(candidates):
        raise ValueError('Incomplete selection grid')
    summaries = []
    baseline = {r['seed']:r for r in rows if r['recipe_id'] == 'frozen'}
    if set(baseline) != set(seeds):
        raise ValueError('Missing stable reference seeds')
    for cfg in candidates:
        group = [r for r in rows if r['recipe_id'] == cfg['id']]
        if len(group) != len(seeds) or sorted(r['seed'] for r in group) != sorted(seeds):
            raise ValueError('Missing or duplicate candidate seed')
        if not all(np.isfinite(r[k]) for r in group for k in ('auc', 'ap', 'criterion')):
            raise ValueError('Invalid inner metrics')
        means = {k:float(np.mean([r[k] for r in group])) for k in ('auc', 'ap', 'criterion')}
        delta = {k:float(np.mean([r[k]-baseline[r['seed']][k] for r in group])) for k in ('auc', 'ap')}
        positives = {k:sum(r[k] > baseline[r['seed']][k] for r in group) for k in ('auc', 'ap')}
        eligible = cfg['id'] == 'frozen' or all(delta[k] > 0 and positives[k] >= min(2, len(seeds)) for k in delta)
        summaries.append(dict(recipe=cfg, **means, mean_deltas=delta,
                              positive_seeds=positives, eligible=eligible))
    eligible = [v for v in summaries if v['eligible']]
    chosen = min(eligible, key=lambda r:(-r['criterion'], r['recipe']['alpha'], r['recipe']['id']))
    return chosen, summaries
