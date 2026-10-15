"""Training-graph applicability survey, not a performance or ceiling estimate."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
CLEAN = HERE.parent/'midrug_mirna_only_20261006'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compress(keys):
    ids = {k: i for i, k in enumerate(sorted(set(keys)))}
    return np.array([ids[k] for k in keys], dtype=np.int64)


def refine(neighbors, initial):
    colors = initial.copy(); history = [colors]
    for _ in range(len(colors)):
        new = compress([(int(colors[i]), tuple(sorted(int(colors[j]) for j in ns))) for i, ns in enumerate(neighbors)])
        assert len(np.unique(new)) >= len(np.unique(colors))
        history.append(new)
        if len(np.unique(new)) == len(np.unique(colors)):
            return history
        colors = new
    raise AssertionError('WL did not stabilize')


def stats(colors, pairs):
    nodes = Counter(colors.tolist()); edge = Counter(map(tuple, colors[pairs].tolist()))
    return dict(node_classes=len(nodes), nodes_in_nonsingleton_classes=sum(v for v in nodes.values() if v > 1),
        largest_node_class=max(nodes.values()), edge_signature_classes=len(edge),
        edges_in_nonsingleton_classes=sum(v for v in edge.values() if v > 1),
        edge_nonsingleton_fraction=sum(v for v in edge.values() if v > 1)/len(pairs))


def graph(n, pairs):
    neighbors = [[] for _ in range(n)]
    for u, v in pairs:
        neighbors[u].append(v); neighbors[v].append(u)
    return neighbors


def main():
    plan = dict(scope='Original training only; topology survey, no model fit or prediction', folds=[0, 1, 2],
        initializations=['node type', 'node type plus exact original attribute row'],
        refinement='Retain own color and sorted multiset of neighbor colors to stabilization; no hash collisions',
        positive_edge_summary='Non-singleton ordered endpoint signatures on observed positive training graph, gamma=1',
        query_deletions='First six positive training records, structural partition change only',
        no_lncrna=True, outer_validation_evaluated=False, test_evaluated=False,
        graph_sha256=sha(CLEAN/'graph.pt'), data_sha256=sha(CLEAN/'inner_data.npz'), source_sha256=sha(__file__))
    pp = HERE/'orbit_precheck_plan.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == plan
    else:
        pp.write_text(json.dumps(plan, indent=2))
    manifest = json.loads((CLEAN/'manifest.json').read_text())
    assert plan['graph_sha256'] == manifest['graph_sha256'] and plan['data_sha256'] == manifest['data_sha256']
    g = torch.load(CLEAN/'graph.pt', map_location='cpu')
    assert set(g.node_types) == {'miRNA', 'drug'} and all('lnc' not in str(e).lower() for e in g.edge_types)
    xs = {k: g[k].x.numpy() for k in ['miRNA', 'drug']}
    sizes = {k: len(v) for k, v in xs.items()}; nm = sizes['miRNA']; n = sum(sizes.values())
    initial = np.concatenate([np.zeros(nm, dtype=int), np.ones(sizes['drug'], dtype=int)])
    # np.unique compares exact numeric rows; no approximate threshold is chosen.
    attribute_initial = []
    raw = {}
    for t, x in xs.items():
        unique, inv, counts = np.unique(x, axis=0, return_inverse=True, return_counts=True)
        raw[t] = dict(nodes=len(x), features=x.shape[1], unique_attribute_rows=len(unique),
            duplicate_groups=int((counts > 1).sum()), nodes_in_duplicate_groups=int(counts[counts > 1].sum()))
        attribute_initial.extend((t, int(i)) for i in inv)
    attribute_initial = compress(attribute_initial)
    records = []
    with np.load(CLEAN/'inner_data.npz') as z:
        for fold in range(3):
            idx = z[f'train_{fold}']; pairs = z['pairs'][idx].copy(); y = z['y'][idx]
            positive = pairs[y == 1]; positive[:, 1] += nm
            ns = graph(n, positive); topology = refine(ns, initial); attributes = refine(ns, attribute_initial)
            final = topology[-1]
            assert not np.any(final[positive[:, 0]] == final[positive[:, 1]])
            assert all(not (set(ns[u]) & set(ns[v])) for u, v in positive)
            perm = np.random.default_rng(20261007+fold).permutation(n)
            shuffled_initial = np.empty(n, dtype=int); shuffled_initial[perm] = initial
            sh = refine(graph(n, perm[positive]), shuffled_initial)[-1]
            np.testing.assert_array_equal(sh[perm], final)
            deletions = []
            for j in range(min(6, len(positive))):
                remaining = np.delete(positive, j, axis=0)
                deleted = refine(graph(n, remaining), initial)[-1]
                old_eq = final[:, None] == final[None, :]; new_eq = deleted[:, None] == deleted[None, :]
                deletions.append(dict(query=positive[j].tolist(), query_includes_drug_offset=nm,
                    distinct_node_pairs_with_changed_comembership=int(np.triu(old_eq != new_eq, 1).sum()),
                    positive_edge_stats=stats(deleted, remaining)))
            records.append(dict(fold=fold, training_records=len(pairs), positive_edges=len(positive),
                type_seeded_rounds=[stats(c, positive) for c in topology],
                attribute_seeded_rounds=[stats(c, positive) for c in attributes],
                no_same_orbit_cross_type_neighbors=True, no_common_one_hop_neighbors_for_target_edges=True,
                permutation_check_passed=True, query_deletions=deletions))
    # Same typed topology does not force equal representations with distinct x.
    small = np.array([[0, 2], [0, 3], [1, 2], [1, 3]])
    colors = refine(graph(4, small), np.array([0, 0, 1, 1]))[-1]
    features = np.array([0., 1., 0., 2.]); score = features[small].sum(axis=1)
    assert len(set(map(tuple, colors[small]))) == 1 and len(np.unique(score)) == 4
    result = dict(status='passed', plan_sha256=sha(pp), original_attribute_rows=raw, records=records,
        synthetic_counterexample=dict(all_four_edges_same_topology_signature=True, equivariant_attribute_sum_scores=score.tolist()),
        no_lncrna=True, no_fits=True, no_model_metric_evaluation=True, outer_validation_evaluated=False, test_evaluated=False)
    (HERE/'orbit_precheck.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(raw=raw, summary=[dict(fold=r['fold'], topology=r['type_seeded_rounds'][-1],
        attributes=r['attribute_seeded_rounds'][-1]) for r in records]), indent=2))


if __name__ == '__main__':
    main()
