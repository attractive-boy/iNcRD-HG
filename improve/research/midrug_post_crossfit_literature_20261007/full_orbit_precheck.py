"""Complement the target-only survey with every original typed graph relation."""
import json
import numpy as np
import torch
from orbit_precheck import HERE, CLEAN, sha, compress, stats


def refine(n, edges, initial):
    ns = [[] for _ in range(n)]
    for s, relation, d in edges:
        ns[d].append((relation, s))
    colors = initial.copy(); history = [colors]
    for _ in range(n):
        new = compress([(int(colors[i]), tuple(sorted((r, int(colors[j])) for r, j in ns[i]))) for i in range(n)])
        assert len(np.unique(new)) >= len(np.unique(colors))
        history.append(new)
        if len(np.unique(new)) == len(np.unique(colors)):
            return history, ns
        colors = new
    raise AssertionError('WL did not stabilize')


def main():
    plan = dict(scope='Supplement target-only orbit_precheck with MiMi, DrugDrug, MiDrug and rev_MiDrug; no fits or predictions',
        graph_sha256=sha(CLEAN/'graph.pt'), data_sha256=sha(CLEAN/'inner_data.npz'), source_sha256=sha(__file__),
        helper_sha256=sha(HERE/'orbit_precheck.py'), target_only_precheck_sha256=sha(HERE/'orbit_precheck.json'),
        folds=[0, 1, 2], initializations=['type', 'type plus exact attributes'],
        refinement='Directed incoming relation-aware color refinement to stabilization',
        query_deletions='First six positive training pairs; remove both target directions before all refinement',
        no_lncrna=True, outer_validation_evaluated=False, test_evaluated=False)
    pp = HERE/'full_orbit_precheck_plan.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == plan
    else:
        pp.write_text(json.dumps(plan, indent=2))
    manifest = json.loads((CLEAN/'manifest.json').read_text())
    assert sha(CLEAN/'graph.pt') == manifest['graph_sha256'] and sha(CLEAN/'inner_data.npz') == manifest['data_sha256']
    g = torch.load(CLEAN/'graph.pt', map_location='cpu'); assert set(g.node_types) == {'miRNA', 'drug'}
    assert set(g.edge_types) == {('miRNA','MiMi','miRNA'), ('drug','DrugDrug','drug'),
        ('miRNA','MiDrug','drug'), ('drug','rev_MiDrug','miRNA')}
    nm, nd = len(g['miRNA'].x), len(g['drug'].x); n = nm+nd
    offsets = {'miRNA': 0, 'drug': nm}; initial = np.r_[np.zeros(nm, dtype=int), np.ones(nd, dtype=int)]
    attr_keys = []
    for t in ['miRNA', 'drug']:
        _, inv = np.unique(g[t].x.numpy(), axis=0, return_inverse=True)
        attr_keys.extend((t, int(i)) for i in inv)
    attr = compress(attr_keys); records = []
    fixed = []
    for s, rel, d in g.edge_types:
        if s == d:
            fixed.extend((int(u)+offsets[s], rel, int(v)+offsets[d]) for u, v in g[s, rel, d].edge_index.T)
    with np.load(CLEAN/'inner_data.npz') as z:
        for fold in range(3):
            idx = z[f'train_{fold}']; pairs = z['pairs'][idx]; y = z['y'][idx]
            positive = pairs[y == 1].copy(); positive[:, 1] += nm
            target = [(int(u), 'MiDrug', int(v)) for u, v in positive]
            reverse = [(int(v), 'rev_MiDrug', int(u)) for u, v in positive]
            edges = fixed+target+reverse
            h, ns = refine(n, edges, initial); ha, _ = refine(n, edges, attr); final = h[-1]
            perm = np.random.default_rng(20261007+fold).permutation(n)
            perm_init = np.empty(n, dtype=int); perm_init[perm] = initial
            hp, _ = refine(n, [(int(perm[s]), r, int(perm[d])) for s, r, d in edges], perm_init)
            np.testing.assert_array_equal(hp[-1][perm], final)
            near = [set(j for _, j in rows) for rows in ns]
            deletions = []
            for j in range(min(6, len(positive))):
                removed = {target[j], reverse[j]}; masked = [e for e in edges if e not in removed]
                assert len(edges)-len(masked) == 2
                hd, _ = refine(n, masked, initial); deleted = hd[-1]
                changed = (deleted[:, None] == deleted[None, :]) != (final[:, None] == final[None, :])
                deletions.append(dict(training_positive_index=j, changed_comembership_pairs=int(np.triu(changed, 1).sum()),
                    final_node_classes=len(np.unique(deleted))))
            records.append(dict(fold=fold, graph_directed_edges=len(edges),
                type_seeded_rounds=[stats(c, positive) for c in h], attribute_seeded_rounds=[stats(c, positive) for c in ha],
                target_positive_pairs_with_shared_incoming_neighbors=sum(bool(near[u] & near[v]) for u, v in positive),
                directed_graph_edges_between_equal_final_colors=sum(final[s] == final[d] for s, _, d in edges).item(),
                permutation_check_passed=True, query_deletions=deletions))
    out = dict(status='passed', plan_sha256=sha(pp), records=records, no_lncrna=True, no_fits=True,
        no_model_metric_evaluation=True, outer_validation_evaluated=False, test_evaluated=False,
        interpretation='Relation-aware structural partitions are descriptive; neither they nor unique input rows guarantee distinct learned embeddings or accurate predictions.')
    (HERE/'full_orbit_precheck.json').write_text(json.dumps(out, indent=2))
    print(json.dumps([dict(fold=r['fold'], full_topology=r['type_seeded_rounds'][-1],
        shared_incoming=r['target_positive_pairs_with_shared_incoming_neighbors']) for r in records], indent=2))


if __name__ == '__main__':
    main()
