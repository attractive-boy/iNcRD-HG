"""Shared semantic input adaptation and projection of actual parameter updates.

The adapter runs after support-fitted preprocessing and before column encoding.
Feature filtering and shuffling are explicitly inverted; no installed source is
modified and no task identifier is an input to the adapter.
"""
from contextlib import contextmanager
from itertools import combinations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class SharedInputAdapter(nn.Module):
    def __init__(self, rank=8, shared_features=31, total_features=36):
        super().__init__()
        self.rank = rank
        self.shared_features = shared_features
        self.total_features = total_features
        self.down = nn.Linear(shared_features, rank, bias=False)
        self.up = nn.Linear(rank, shared_features, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        if x.shape[-1] != self.total_features:
            raise ValueError('Expected the canonical 36-column schema')
        core = x[..., :self.shared_features]
        return torch.cat((core + self.up(torch.tanh(self.down(core))),
                          x[..., self.shared_features:]), dim=-1)

    def transformed(self, x, retained, shuffles=None):
        """Map filtered/shuffled columns to their canonical semantic positions."""
        retained = torch.as_tensor(retained, dtype=torch.long, device=x.device)
        if x.ndim != 3 or retained.numel() != x.shape[-1]:
            raise ValueError('Feature mask does not match the model input')
        if (retained < 0).any() or (retained >= self.total_features).any() or retained.unique().numel() != retained.numel():
            raise ValueError('Invalid retained feature indices')
        if shuffles is not None:
            order = torch.as_tensor(shuffles, dtype=torch.long, device=x.device)
            if order.shape != (x.shape[0], x.shape[2]):
                raise ValueError('One feature permutation per member is required')
            expected = torch.arange(x.shape[2], device=x.device).expand_as(order)
            if not torch.equal(order.sort(-1).values, expected):
                raise ValueError('Feature shuffles must be permutations')
            inverse = order.argsort(-1)
            canonical = x.gather(-1, inverse[:, None].expand_as(x))
        else:
            canonical = x
        full = x.new_zeros(*x.shape[:-1], self.total_features)
        full = full.index_copy(-1, retained, canonical)
        adapted = self(full).index_select(-1, retained)
        return adapted if shuffles is None else adapted.gather(-1, order[:, None].expand_as(x))


def retained_features(clf):
    mask = np.asarray(clf.ensemble_generator_.unique_filter_.features_to_keep_, dtype=bool)
    if mask.shape != (36,):
        raise ValueError('Numeric preprocessing changed the canonical schema')
    return np.flatnonzero(mask)


@contextmanager
def install_adapter(clf, adapter):
    """Temporary hook; cached column/row/KV representations are not permitted."""
    if clf.kv_cache or getattr(clf, 'model_kv_cache_', None) is not None:
        raise ValueError('Input adaptation requires recomputing all representations')
    retained = retained_features(clf)

    def before_forward(model, args, kwargs):
        if args:
            raise ValueError('Expected the official keyword-based prediction call')
        kwargs = dict(kwargs)
        kwargs['X'] = adapter.transformed(kwargs['X'], retained, kwargs.get('feature_shuffles'))
        return args, kwargs

    handle = clf.model_.register_forward_pre_hook(before_forward, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def prepare_episode(x, y, support, query, *, seed, members=2):
    """Official support-only preprocessing, with explicit class/column maps."""
    from tabicl._sklearn.preprocessing import EnsembleGenerator
    support, query = np.asarray(support), np.asarray(query)
    if len(np.intersect1d(support, query)):
        raise ValueError('Support and pseudo-query overlap')
    generator = EnsembleGenerator(classification=True, n_estimators=members,
        norm_methods=['none', 'power'], feat_shuffle_method='latin',
        class_shuffle_method='shift', outlier_threshold=4., random_state=seed)
    generator.fit(x[support], y[support].astype(np.int64))
    retained = np.flatnonzero(generator.unique_filter_.features_to_keep_)
    transformed = generator.transform(x[query], mode='both')
    xs, ys, columns, classes = [], [], [], []
    for norm, (xx, yy) in transformed.items():
        xs.append(xx)
        ys.append(yy)
        columns.extend(generator.feature_shuffles_[norm])
        classes.extend(generator.class_shuffles_[norm])
    return dict(X=torch.as_tensor(np.concatenate(xs), dtype=torch.float32),
        ys=torch.as_tensor(np.concatenate(ys), dtype=torch.float32),
        yq=torch.as_tensor(y[query], dtype=torch.long),
        columns=torch.as_tensor(columns, dtype=torch.long),
        classes=torch.as_tensor(classes, dtype=torch.long),
        retained=torch.as_tensor(retained, dtype=torch.long),
        support=torch.as_tensor(support, dtype=torch.long),
        query=torch.as_tensor(query, dtype=torch.long))


def sample_episode(labels, rows, support_size, query_size, rng):
    """Disjoint stratified draws, preserving fold prevalence up to rounding."""
    labels, rows = np.asarray(labels), np.asarray(rows)
    prevalence = float(labels[rows].mean())
    positive_support = int(np.clip(round(prevalence * support_size), 1, support_size - 1))
    positive_query = int(np.clip(round(prevalence * query_size), 1, query_size - 1))
    support, query = [], []
    for label, ns, nq in [(1, positive_support, positive_query),
                          (0, support_size - positive_support, query_size - positive_query)]:
        pool = rng.permutation(rows[labels[rows] == label])
        if len(pool) < ns + nq:
            raise ValueError('Not enough fitting rows for a disjoint stratified episode')
        support.extend(pool[:ns])
        query.extend(pool[ns:ns + nq])
    return rng.permutation(support), rng.permutation(query)


def episode_margin(model, adapter, episode, device):
    """Average class-corrected logits, retaining input gradients through backbone."""
    x = episode['X'].to(device)
    ys = episode['ys'].to(device)
    adapted = adapter.transformed(x, episode['retained'], episode['columns'])
    # Frozen encoders still need the differentiable training forward path.
    model.train()
    with sdpa_kernel(SDPBackend.MATH):
        logits = model(adapted, ys)[..., :2]
    classes = episode['classes'].to(device)
    logits = logits.gather(-1, classes[:, None].expand_as(logits))
    average = logits.mean(0)
    return (average[:, 1] - average[:, 0]) / .9, episode['yq'].to(device)


def ranking_losses(margin, labels):
    """Global smooth AUC and AP surrogates, computed in double precision.

The AP surrogate uses all pseudo-queries, including cross-episode comparisons
when their margins are concatenated. Neither loss is the evaluation metric.
"""
    score = margin.double()
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        raise ValueError('Both classes are needed for ranking losses')
    auc = F.softplus(score[negative][None] - score[positive][:, None]).mean()
    compare = torch.sigmoid(score[None, :] - score[positive, None])
    # Exclude a positive's self-comparison (sigmoid(0) = 1/2).
    rank = .5 + compare.sum(1)
    positive_rank = .5 + compare[:, positive].sum(1)
    ap = 1. - (positive_rank / rank).mean()
    return torch.stack((auc, ap))


def objective(margin, labels):
    return F.binary_cross_entropy_with_logits(margin.double(), labels.double()) + ranking_losses(margin, labels).sum()


def parameter_vector(module):
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])


def set_parameter_vector(module, value):
    offset = 0
    with torch.no_grad():
        for parameter in module.parameters():
            size = parameter.numel()
            parameter.copy_(value[offset:offset + size].reshape_as(parameter))
            offset += size
    if offset != value.numel():
        raise ValueError('Wrong parameter vector size')


def flatten_gradient(loss, module, *, retain_graph=False):
    values = torch.autograd.grad(loss, tuple(module.parameters()), retain_graph=retain_graph)
    result = torch.cat([v.reshape(-1) for v in values])
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite adapter gradient')
    return result


def project_update(proposed, gradients):
    """Exact active-set projection for at most two homogeneous half-spaces.

The input is the actual proposed parameter change, including Adam momentum and
preconditioning. Sign convention: theta_next = theta + update, so G @ update <= 0.
Double precision and row normalization make the tiny QP well conditioned.
"""
    original = proposed.detach().double().cpu().numpy()
    matrix = gradients.detach().double().cpu().numpy()
    if matrix.ndim != 2 or matrix.shape[1] != len(original) or len(matrix) > 2:
        raise ValueError('Expected at most two ranking constraints')
    if not np.isfinite(original).all() or not np.isfinite(matrix).all():
        raise FloatingPointError('Nonfinite projection input')
    norms = np.linalg.norm(matrix, axis=1)
    matrix = matrix[norms > 1e-14] / norms[norms > 1e-14, None]
    tolerance = 1e-12 * max(1., np.linalg.norm(original))
    candidates = [np.zeros_like(original)]
    if np.all(matrix @ original <= tolerance):
        candidates.append(original)
    for count in range(1, len(matrix) + 1):
        for indices in combinations(range(len(matrix)), count):
            active = matrix[list(indices)]
            multipliers = np.linalg.lstsq(active @ active.T, active @ original, rcond=1e-12)[0]
            candidate = original - active.T @ multipliers
            if np.min(multipliers) >= -tolerance and np.all(matrix @ candidate <= tolerance):
                candidates.append(candidate)
    best = min(candidates, key=lambda value: np.sum((value - original) ** 2))
    return torch.as_tensor(best, dtype=proposed.dtype, device=proposed.device)


def save_adapter(path, adapter, metadata):
    torch.save(dict(state={k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                    metadata=metadata, rank=adapter.rank), path)


def load_adapter(path, device):
    state = torch.load(path, map_location='cpu', weights_only=True)
    adapter = SharedInputAdapter(rank=state['rank'])
    adapter.load_state_dict(state['state'], strict=True)
    return adapter.to(device).eval(), state['metadata']
