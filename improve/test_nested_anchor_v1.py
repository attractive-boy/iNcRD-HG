"""Query-label and graph reconstruction boundaries of nested residual offsets."""
import inspect
import unittest

import numpy as np

from improve import nested_anchor_inputs_v1 as nested


def fixture():
    counts = np.array([6, 8, 6, 3])
    names = ['lncRNA', 'miRNA', 'drug', 'circRNA']
    relations = [('lncRNA', 'LncLnc', 'lncRNA'), ('lncRNA', 'LncDrug', 'drug'),
        ('miRNA', 'MiMi', 'miRNA'), ('miRNA', 'MiDrug', 'drug'),
        ('drug', 'DrugDrug', 'drug'), ('lncRNA', 'LncMi', 'miRNA'),
        ('circRNA', 'CircCirc', 'circRNA'), ('circRNA', 'CircDrug', 'drug'),
        ('circRNA', 'CircHost', 'circRNA')]
    offsets = dict(zip(names, np.r_[0, np.cumsum(counts)[:-1]]))
    nc = dict(zip(names, counts))
    pairs = np.indices((8, 6)).reshape(2, -1)
    y = ((pairs[0] + pairs[1]) % 2).astype(np.float32)
    es, rs, codes = [], [], []
    for i, (src, rel, dst) in enumerate(relations):
        if rel == 'MiDrug':
            e = pairs[:, y == 1]
        elif src == dst:
            e = np.array([np.arange(nc[src] - 1), np.arange(1, nc[src])])
        else:
            k = min(nc[src], nc[dst])
            e = np.array([np.arange(k), np.arange(k)])
        es.append(e + np.array([offsets[src], offsets[dst]])[:, None])
        rs.append(np.full(e.shape[1], i))
        codes.append(e[0] * 6 + e[1] if rel == 'MiDrug' else np.full(e.shape[1], -1))
    a = dict(pairs_train=pairs, y_train=y, folds=np.arange(len(y)) % 5,
        node_counts=counts, pair_offset=np.array([6, 14]), nd=np.array(6),
        edges=np.concatenate(es, 1), relation=np.concatenate(rs), target_codes=np.concatenate(codes))
    rng = np.random.default_rng(9241)
    for i, n in enumerate(counts):
        a['x_' + str(i)] = rng.normal(size=(n, 4)).astype(np.float32)
    return a, dict(node_types=names, physical_relations=relations)


class NestedAnchorBoundary(unittest.TestCase):
    def test_physical_support_and_reverse_arcs(self):
        a, desc = fixture()
        selected = (a['folds'] != 0) & (a['y_train'] == 1)
        g = nested.FeatureGraph(a, desc, a['pairs_train'][:, selected])
        target = g['miRNA', 'MiDrug', 'drug'].edge_index.numpy()
        np.testing.assert_array_equal(target, a['pairs_train'][:, selected])
        np.testing.assert_array_equal(g['drug', 'rev_MiDrug', 'miRNA'].edge_index.numpy(), target[::-1])
        self.assertEqual(list(g.edge_index_dict)[:9], desc['physical_relations'])
        edge = g['miRNA', 'MiMi', 'miRNA'].edge_index.numpy()
        self.assertEqual(set(map(tuple, edge.T)), set(map(tuple, edge[::-1].T)))

    def test_nested_query_order_and_scaler(self):
        a, desc = fixture()
        mask = a['folds'] != 0
        ps, ys, pq = a['pairs_train'][:, mask], a['y_train'][mask], a['pairs_train'][:, ~mask]
        graph = nested.FeatureGraph(a, desc, ps[:, ys == 1])
        x, transform, audit = nested.build_nested(graph, 'MiDrug', ps, ys, pq)
        rev, _, _ = nested.build_nested(graph, 'MiDrug', ps, ys, pq[:, ::-1])
        np.testing.assert_array_equal(x['raw_train'], rev['raw_train'])
        np.testing.assert_array_equal(x['raw_query'], rev['raw_query'][::-1])
        np.testing.assert_array_equal(x['x_query'], rev['x_query'][::-1])
        np.testing.assert_allclose(transform.transform(x['raw_train']), x['x_train'], atol=1e-6)
        self.assertTrue(all(f['forbidden_fact_overlap'] == 0 for f in audit['folds']))
        self.assertEqual(x['raw_train'].shape[1], 36)
        self.assertEqual(x['raw_query_views'].shape[0], 5)

    def test_outer_query_labels_cannot_affect_inputs(self):
        a, desc = fixture()
        support = a['folds'] != 0
        def build(arrays):
            pairs, y = arrays['pairs_train'][:, support], arrays['y_train'][support]
            graph = nested.FeatureGraph(arrays, desc, pairs[:, y == 1])
            return nested.build_nested(graph, 'MiDrug', pairs, y, arrays['pairs_train'][:, ~support])[0]
        expected = build(a)
        changed = {k: v.copy() for k, v in a.items()}
        changed['y_train'][~support] = 1 - changed['y_train'][~support]
        actual = build(changed)
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])
        signature = inspect.signature(nested.build_nested)
        self.assertNotIn('query_labels', signature.parameters)
        self.assertFalse(any(k.startswith('y_') and k != 'y_train' for k in actual))

    def test_query_overlap_and_contaminated_graph_rejected(self):
        a, desc = fixture()
        mask = a['folds'] != 0
        pairs, y, query = a['pairs_train'][:, mask], a['y_train'][mask], a['pairs_train'][:, ~mask]
        graph = nested.FeatureGraph(a, desc, pairs[:, y == 1])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            nested.build_nested(graph, 'MiDrug', pairs, y, pairs[:, :2])
        contaminated = nested.FeatureGraph(a, desc, a['pairs_train'][:, a['y_train'] == 1])
        with self.assertRaisesRegex(ValueError, 'exactly support'):
            nested.build_nested(contaminated, 'MiDrug', pairs, y, query)


if __name__ == '__main__':
    unittest.main()
