"""Bounded, typed structural programs and conditional degree-null features.

This is an enumerated grammar with supervised feature selection, not an LLM
code-evolution reproduction. All label-dependent inputs are explicit support
edges. No node identifier is ever a numerical feature.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression


SIGNATURES = {'S': ('r', 'r'), 'T': ('o', 'o'), 'D': ('d', 'd'),
              'C': ('r', 'o'), 'c': ('o', 'r'), 'A': ('r', 'd'),
              'a': ('d', 'r'), 'B': ('o', 'd'), 'b': ('d', 'o')}
INVERSE = {'C': 'c', 'c': 'C', 'A': 'a', 'a': 'A', 'B': 'b', 'b': 'B'}
EXISTING_COUNTS = {('S', 'A'), ('A', 'D'), ('C', 'B'), ('S', 'A', 'D'),
                   ('S', 'S', 'A'), ('A', 'D', 'D'), ('A', 'a', 'A')}


@dataclass(frozen=True)
class Program:
    path: tuple
    mode: str = 'count'

    def __post_init__(self):
        if self.mode not in ('count', 'walk', 'symmetric', 'nonbacktracking'):
            raise ValueError('Unknown program mode')
        node = 'r'
        for symbol in self.path:
            if symbol not in SIGNATURES or SIGNATURES[symbol][0] != node:
                raise ValueError('Ill-typed path: ' + str(self.path))
            node = SIGNATURES[symbol][1]
        if node != 'd':
            raise ValueError('A program must return RNA-by-drug values')
        if self.mode == 'nonbacktracking' and self.path != ('A', 'a', 'A', 'a', 'A'):
            raise ValueError('Only target-relation length-five nonbacktracking is implemented')

    @property
    def name(self):
        return self.mode + ':' + '.'.join(self.path)

    def record(self):
        return dict(name=self.name, path=list(self.path), mode=self.mode, length=len(self.path))


def grammar(max_length=4):
    result = []

    def extend(path, node):
        if len(path) >= 2 and node == 'd':
            for mode in ('count', 'walk', 'symmetric'):
                if mode == 'count' and path in EXISTING_COUNTS:
                    continue
                result.append(Program(path, mode))
        if len(path) == max_length:
            return
        for symbol, (src, dst) in sorted(SIGNATURES.items()):
            if src == node and (not path or INVERSE.get(path[-1]) != symbol):
                extend(path + (symbol,), dst)

    extend((), 'r')
    result.append(Program(('A', 'a', 'A', 'a', 'A'), 'nonbacktracking'))
    return sorted(result, key=lambda x: (len(x.path), x.name))


def normalize(matrix, mode):
    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    if mode == 'count':
        return matrix
    row = np.asarray(matrix.sum(1)).ravel()
    if mode == 'walk':
        return sparse.diags(1. / np.maximum(row, 1.)) @ matrix
    col = np.asarray(matrix.sum(0)).ravel()
    return sparse.diags(1. / np.sqrt(np.maximum(row, 1.))) @ matrix @ sparse.diags(1. / np.sqrt(np.maximum(col, 1.)))


def nonbacktracking_five(a):
    """Exact counts of length-five walks with no immediate node return."""
    a = sparse.csr_matrix(a, dtype=np.float64)
    nr, nd = a.shape
    adjacency = sparse.bmat([[None, a], [a.T, None]], format='csr')
    identity = sparse.eye(nr + nd, format='csr')
    degree = sparse.diags(np.asarray(adjacency.sum(1)).ravel())
    previous = adjacency
    current = adjacency @ adjacency - degree
    for _ in range(3, 6):
        previous, current = current, adjacency @ current - (degree - identity) @ previous
    values = current[:nr, nr:].toarray()
    if values.min(initial=0.) < -1e-7:
        raise FloatingPointError('Negative nonbacktracking count')
    return np.maximum(values, 0.)


def degree_swaps(support, shape, forbidden, seed, swaps_per_edge=5, max_attempt_factor=40):
    """Double-edge swaps preserve both degree sequences and exclude all queries.

    Only the target relation is randomized. Side relations stay fixed. The
    forbidden set uses query identities without labels; both classes obey the
    same constraint. Returned diagnostics expose immobile/poorly mixed graphs.
    """
    edges = np.asarray(support, dtype=np.int64).copy()
    forbidden = np.asarray(forbidden, dtype=np.int64)
    nr, nd = shape
    matrix = np.zeros(shape, dtype=np.uint8)
    blocked = np.zeros(shape, dtype=bool)
    blocked[forbidden[0], forbidden[1]] = True
    matrix[edges[0], edges[1]] = 1
    if matrix.sum() != edges.shape[1] or np.any(blocked[edges[0], edges[1]]):
        raise ValueError('Duplicate or query-positive support')
    before_r, before_d = matrix.sum(1), matrix.sum(0)
    original = matrix.copy()
    goal = swaps_per_edge * edges.shape[1]
    rng = np.random.RandomState(seed)
    successes, attempts = 0, 0
    limit = max_attempt_factor * max(goal, 1)
    while successes < goal and attempts < limit and edges.shape[1] >= 2:
        i, j = rng.randint(edges.shape[1], size=2)
        attempts += 1
        r1, d1 = edges[:, i]
        r2, d2 = edges[:, j]
        if r1 == r2 or d1 == d2 or matrix[r1, d2] or matrix[r2, d1] or blocked[r1, d2] or blocked[r2, d1]:
            continue
        matrix[r1, d1] = matrix[r2, d2] = 0
        matrix[r1, d2] = matrix[r2, d1] = 1
        edges[1, i], edges[1, j] = d2, d1
        successes += 1
    np.testing.assert_array_equal(before_r, matrix.sum(1))
    np.testing.assert_array_equal(before_d, matrix.sum(0))
    if np.any(matrix[forbidden[0], forbidden[1]]):
        raise AssertionError('Null graph includes a forbidden query')
    return edges, dict(seed=int(seed), edges=int(edges.shape[1]), accepted=successes,
        target=goal, attempts=attempts, completed=successes == goal,
        fraction_edges_changed=float(np.count_nonzero((matrix == 1) & (original == 0)) / max(edges.shape[1], 1)),
        degree_error=0, query_edges=0)


class ProgramEncoder:
    def __init__(self, graph, task):
        rna, other = ('lncRNA', 'miRNA') if task == 'LncDrug' else ('miRNA', 'lncRNA')
        other_task = 'MiDrug' if task == 'LncDrug' else 'LncDrug'
        self.shape = (graph[rna].num_nodes, graph['drug'].num_nodes)

        def matrix(key, diagonal=False):
            e = graph[key].edge_index.cpu().numpy()
            value = sparse.csr_matrix((np.ones(e.shape[1]), e),
                                      shape=(graph[key[0]].num_nodes, graph[key[2]].num_nodes))
            value.data[:] = 1.
            if diagonal:
                value.setdiag(0.)
                value.eliminate_zeros()
            return value

        cross = matrix(('lncRNA', 'LncMi', 'miRNA'))
        cross = cross if rna == 'lncRNA' else cross.T.tocsr()
        other_assoc = matrix((other, other_task, 'drug'))
        self.fixed = {
            'S': matrix((rna, 'LncLnc' if rna == 'lncRNA' else 'MiMi', rna), True),
            'T': matrix((other, 'LncLnc' if other == 'lncRNA' else 'MiMi', other), True),
            'D': matrix(('drug', 'DrugDrug', 'drug'), True),
            'C': cross, 'c': cross.T.tocsr(), 'B': other_assoc, 'b': other_assoc.T.tocsr()}

    def evaluate(self, pairs, support, programs):
        pairs = np.asarray(pairs, dtype=np.int64)
        support = np.asarray(support, dtype=np.int64)
        a = sparse.csr_matrix((np.ones(support.shape[1]), support), shape=self.shape)
        if np.any(np.asarray(a[pairs[0], pairs[1]]).ravel()):
            raise ValueError('Query positive is visible to its own program')
        matrices = dict(self.fixed, A=a, a=a.T.tocsr())
        prepared, cache = {}, {}

        def value(path, mode):
            key = (mode, path)
            if key not in cache:
                symbol = path[0]
                if (mode, symbol) not in prepared:
                    prepared[mode, symbol] = normalize(matrices[symbol], mode)
                mat = prepared[mode, symbol]
                cache[key] = mat.toarray() if len(path) == 1 else mat @ value(path[1:], mode)
            return cache[key]

        columns = []
        for program in programs:
            if program.mode == 'nonbacktracking':
                result = nonbacktracking_five(a)
            else:
                result = value(program.path, program.mode)
            columns.append(result[pairs[0], pairs[1]])
        output = np.stack(columns, 1).astype(np.float32)
        if not np.isfinite(output).all():
            raise FloatingPointError('Non-finite program feature')
        return output

    def with_null(self, pairs, support, programs, null_samples=8, seed=20260911,
                  swaps_per_edge=5):
        real = self.evaluate(pairs, support, programs)
        samples, audit = [], []
        for index in range(null_samples):
            changed, details = degree_swaps(support, self.shape, pairs,
                                            seed + index, swaps_per_edge)
            samples.append(self.evaluate(pairs, changed, programs))
            audit.append(details)
        draws = np.stack(samples).astype(np.float64)
        mean, std = draws.mean(0), draws.std(0)
        # A relative floor avoids division by tiny sampling variance. Constant
        # null features retain their raw value instead of creating fake z-scores.
        floor = np.maximum(.05 * np.maximum(mean, 0.), 1e-6)
        fallback = std <= 1e-8 * np.maximum(mean, 1.)
        excess = np.clip((real - mean) / np.maximum(std, floor), -10., 10.)
        excess = np.where(fallback, real, excess).astype(np.float32)
        return real, excess, dict(draws=audit, fallback_fraction=float(fallback.mean()),
                                  all_swaps_completed=all(x['completed'] for x in audit))


def select_programs(base, candidates, labels, programs, k=12, seed=20260911):
    """Training-only residual screening, complexity and redundancy penalties."""
    base = np.asarray(base, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    z = np.log1p(np.maximum(candidates, 0.))
    scale = z.std(0)
    z = (z - z.mean(0)) / np.maximum(scale, 1e-12)
    nuisance = np.log1p(np.maximum(base[:, :3], 0.))
    fit = LogisticRegression(C=1., max_iter=2000, random_state=seed).fit(nuisance, labels)
    residual = labels - fit.predict_proba(nuisance)[:, 1]
    residual = (residual - residual.mean()) / max(residual.std(), 1e-12)
    association = np.abs(z.T @ residual) / len(labels)
    base_z = np.sign(base) * np.log1p(np.abs(base))
    base_z = (base_z - base_z.mean(0)) / np.maximum(base_z.std(0), 1e-12)
    base_corr = np.max(np.abs(z.T @ base_z) / len(labels), axis=1)
    eligible = np.flatnonzero((scale > 1e-8) & (base_corr < .9999))
    if len(eligible) < k:
        raise ValueError('Too few nonconstant, nonduplicate programs for fixed budget')
    lengths = np.array([len(p.path) for p in programs])
    chosen, selected_scores = [], []
    for _ in range(k):
        redundancy = np.max(np.abs(z.T @ z[:, chosen]) / len(labels), 1) if chosen else np.zeros(len(programs))
        score = association - .002 * (lengths - 2) - .02 * redundancy - .01 * base_corr
        available = [int(i) for i in eligible if i not in chosen]
        winner = min(available, key=lambda i: (-score[i], programs[i].name))
        chosen.append(winner)
        selected_scores.append(float(score[winner]))
    random_ids = np.random.RandomState(seed).choice(eligible, size=k, replace=False).tolist()
    return dict(selected=chosen, random=random_ids, k=k, eligible_count=len(eligible),
        selected_scores=selected_scores,
        program_statistics=[dict(**p.record(), residual_association=float(association[i]),
            base_correlation=float(base_corr[i]), eligible=bool(i in eligible)) for i, p in enumerate(programs)])
