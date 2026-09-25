"""Semantic checks for the pinned pretrained graph-context runtime."""
import unittest
import sys

import numpy as np
import torch

from improve import kgpfn_runtime_v1 as runtime


def literal_rspmm(edges, kinds, weights, relation, values, sum='add', mul='mul'):
    if sum != 'add' or mul != 'mul':
        raise ValueError('Reference only covers sum/DistMult')
    out = torch.zeros_like(values)
    # Independent edge loop follows the row0/col1 ABI in the pinned C++ source.
    for k in range(len(kinds)):
        out[edges[0, k]] += weights[k] * relation[kinds[k]] * values[edges[1, k]]
    return out


class RuntimeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = runtime.load_model('cpu', entity_chunk_size=2)
        cls.graph = runtime.graph_from_edges(
            np.array([[0, 1, 2, 3, 4, 1], [1, 2, 3, 4, 5, 4]]),
            np.array([0, 1, 2, 0, 1, 2]), 7, 3)
        cls.query = np.array([[0, 5, 0], [3, 6, 0], [1, 5, 0]])
        cls.context = np.array([[0, 4, 0], [1, 3, 0], [3, 5, 0], [4, 6, 0]])
        cls.labels = np.array([1., 0., 1., 0.], dtype=np.float32)

    def test_sparse_orientation_weights_duplicates_and_mutation(self):
        torch.manual_seed(821)
        edges = torch.tensor([[0, 1, 0, 2, 0, 4], [1, 2, 1, 0, 3, 2]])
        types = torch.tensor([0, 1, 0, 1, 2, 2])
        weight = torch.tensor([.2, -.4, .6, 1., .8, .3], dtype=torch.float64)
        relation = torch.randn(3, 5, dtype=torch.float64)
        values = torch.randn(6, 5, dtype=torch.float64)
        op = runtime.RelationalSparseSum()
        expected = literal_rspmm(edges, types, weight, relation, values)
        torch.testing.assert_close(op(edges, types, weight, relation, values), expected, atol=1e-12, rtol=1e-12)
        backwards = literal_rspmm(edges.flip(0), types, weight, relation, values)
        self.assertGreater(float((expected - backwards).abs().max()), .1)
        weight[0] += .3
        torch.testing.assert_close(op(edges, types, weight, relation, values),
                                   literal_rspmm(edges, types, weight, relation, values), atol=1e-12, rtol=1e-12)
        with self.assertRaises(ValueError):
            op(edges, types, weight, relation, values, mul='add')
        with self.assertRaises(ValueError):
            op(edges, types, weight, relation, values.clone().requires_grad_())

    def test_strict_state_and_full_model_literal_backend(self):
        model = self.model
        self.assertEqual(len(model.state_dict()), 368)
        self.assertEqual(sum(x.numel() for x in model.parameters()), 26668874)
        self.assertFalse(any(x.requires_grad for x in model.parameters()))
        actual = runtime.score(model, self.graph, self.query, self.context, self.labels)
        module = sys.modules['model.ultra.rspmm']
        saved = module.generalized_rspmm
        try:
            module.generalized_rspmm = literal_rspmm
            expected = runtime.score(model, self.graph, self.query, self.context, self.labels)
        finally:
            module.generalized_rspmm = saved
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)

    def test_query_context_permutation_and_label_effect(self):
        model, graph = self.model, self.graph
        actual = runtime.score(model, graph, self.query, self.context, self.labels)
        reverse = runtime.score(model, graph, self.query[::-1].copy(), self.context, self.labels)[::-1]
        chunks = np.concatenate([runtime.score(model, graph, row[None], self.context, self.labels)
                                 for row in self.query])
        context_order = np.array([2, 0, 3, 1])
        reordered = runtime.score(model, graph, self.query,
                                  self.context[context_order], self.labels[context_order])
        for replay in (reverse, chunks, reordered):
            np.testing.assert_allclose(actual, replay, atol=3e-5, rtol=3e-5)
        changed = runtime.score(model, graph, self.query, self.context, 1. - self.labels)
        self.assertGreater(float(np.abs(changed - actual).max()), 1e-5)

    def test_one_traversal_and_row_cache_preserve_author_forward(self):
        model, graph = self.model, self.graph
        all_triples = np.concatenate((self.context, self.query))
        with torch.no_grad():
            reference, corrected = model._build_structure_aligned(
                graph, torch.as_tensor(all_triples)[None], len(self.context),
                [torch.as_tensor(self.labels)])
            fused, raw = runtime.encode_once(model, graph, all_triples, chunk_size=2)
            torch.testing.assert_close(fused, reference[0], atol=3e-5, rtol=3e-5)
            correction = model._apply_label_correction(raw[:len(self.context)][None],
                                                       [torch.as_tensor(self.labels)])
            torch.testing.assert_close(correction[0], corrected[0], atol=0, rtol=0)
            rows = runtime.encode_rows(model, fused, chunk_size=3)
            cached = runtime.score_rows(model, rows[4:], rows[:4], correction[0], batch_size=2)
            normal = runtime.score(model, graph, self.query, self.context, self.labels)
            np.testing.assert_allclose(cached, normal, atol=3e-5, rtol=3e-5)
            # Verify the actual published feature overwrite, not its comment.
            altered = fused.clone()
            altered[:, :4] += 100.
            same_rows = runtime.encode_rows(model, altered, chunk_size=3)
            torch.testing.assert_close(rows, same_rows, atol=0, rtol=0)

    def test_query_and_context_facts_removed_before_inverse_and_relation_graph(self):
        a = dict(edges=np.array([[0, 3, 4, 3, 1], [1, 5, 6, 6, 5]]),
                 relation=np.array([0, 1, 1, 1, 2]), target_codes=np.array([-1, 0, 3, 1, -1]),
                 pairs_train=np.array([[0, 1, 0, 1], [0, 1, 1, 0]]),
                 y_train=np.array([1., 1., 1., 0.]), folds=np.array([0, 1, 2, 0]),
                 node_counts=np.array([3, 2, 2, 0]), pair_offset=np.array([3, 5]), nd=np.array(2))
        actual = runtime.masked_graph(a, [0, 1], device='cpu')
        for pair in ((3, 5), (5, 3), (4, 6), (6, 4)):
            self.assertFalse(bool((actual.edge_index == torch.tensor(pair)[:, None]).all(0).any()))
        altered = {k: v.copy() for k, v in a.items()}
        altered['y_train'][[0, 1]] = 0.
        keep = ~np.isin(altered['target_codes'], [0, 3])
        for key in ('relation', 'target_codes'):
            altered[key] = altered[key][keep]
        altered['edges'] = altered['edges'][:, keep]
        equivalent = runtime.masked_graph(altered, [0, 1], device='cpu')
        for name in ('edge_index', 'edge_type'):
            torch.testing.assert_close(getattr(actual, name), getattr(equivalent, name), atol=0, rtol=0)
            torch.testing.assert_close(getattr(actual.relation_graph, name),
                                       getattr(equivalent.relation_graph, name), atol=0, rtol=0)
        with self.assertRaises(ValueError):
            runtime.masked_graph(dict(a, y_val=np.array([1.])), [0], device='cpu')


if __name__ == '__main__':
    unittest.main(verbosity=2)
