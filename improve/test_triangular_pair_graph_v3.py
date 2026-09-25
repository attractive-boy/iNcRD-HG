"""Exact query-only final computation, including duplicate and reordered queries."""
import unittest

import torch

from improve import triangular_pair_graph_v1 as reference
from improve import triangular_pair_graph_v3 as sparse
from improve.test_triangular_pair_graph_v1 import explicit_triangle
from improve.test_nested_anchor_v2 import fixture


class QueryTriangleTests(unittest.TestCase):
    def test_query_operator_and_gradients_match_dense_definition(self):
        torch.manual_seed(727)
        inputs = [torch.randn(6, 6, 2, 3, dtype=torch.float64, requires_grad=True) for _ in range(4)]
        source, destination = torch.tensor([2, 5, 1, 2, 0]), torch.tensor([3, 0, 4, 3, 1])
        weight = torch.randn(5, 2, 3, dtype=torch.float64)
        for kind in sparse.KINDS:
            expected = explicit_triangle(*inputs, kind)[source, destination]
            actual = sparse.query_triangle(*inputs, source, destination, kind)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            d1 = torch.autograd.grad((actual * weight).sum(), inputs)
            d2 = torch.autograd.grad((expected * weight).sum(), inputs)
            for x, y in zip(d1, d2):
                torch.testing.assert_close(x, y, atol=3e-12, rtol=3e-12)

    def test_model_parameters_values_gradients_replay_with_one_and_two_layers(self):
        a, _ = fixture()
        for depth in (1, 2):
            p = sparse.parameters(); p.update(width=8, heads=2, depth=depth, dropout=0., query_chunk=3)
            for kind in sparse.KINDS:
                torch.manual_seed(852)
                model, xs, graphs, _ = reference.setup(a, p, kind, 'cpu')
                torch.nn.init.normal_(model.readout[-1].weight, std=.1)
                replacement, _, _, _ = sparse.setup(a, p, kind, 'cpu')
                replacement.load_state_dict(model.state_dict(), strict=True)
                self.assertEqual(set(model.state_dict()), set(replacement.state_dict()))
                model.train(); replacement.train()
                queries = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None])[:, :11]
                expected = model(xs, graphs[0], queries)
                actual = replacement(xs, graphs[0], queries)
                torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-7)
                actual.sum().backward(); expected.sum().backward()
                for x, y in zip(model.parameters(), replacement.parameters()):
                    torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=2e-5)
                replacement.eval()
                reversed_value = replacement(xs, graphs[0], queries.flip(1)).flip(0)
                chunks = torch.cat([replacement(xs, graphs[0], queries[:, i:i+4])
                                    for i in range(0, queries.shape[1], 4)])
                torch.testing.assert_close(actual, reversed_value, atol=2e-7, rtol=2e-7)
                torch.testing.assert_close(actual, chunks, atol=2e-7, rtol=2e-7)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
