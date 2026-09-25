"""Algebraic and backward equivalence of the contiguous reduction."""
from copy import deepcopy
import unittest

import torch

from improve import triangular_pair_graph_v1 as reference
from improve import triangular_pair_graph_v2 as compiled
from improve.test_triangular_pair_graph_v1 import explicit_triangle
from improve.test_nested_anchor_v2 import fixture


class CompiledTriangleTests(unittest.TestCase):
    def test_contiguous_definition_and_gradients(self):
        torch.manual_seed(922)
        inputs = [torch.randn(5, 5, 2, 3, dtype=torch.float64, requires_grad=True) for _ in range(4)]
        w = torch.randn_like(inputs[0])
        for kind in compiled.KINDS:
            actual = compiled.contiguous_triangle(*inputs, kind)
            expected = explicit_triangle(*inputs, kind)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            d1 = torch.autograd.grad((actual * w).sum(), inputs)
            d2 = torch.autograd.grad((expected * w).sum(), inputs)
            for x, y in zip(d1, d2):
                torch.testing.assert_close(x, y, atol=2e-12, rtol=2e-12)

    def test_state_dict_and_cpu_model_equivalence(self):
        a, _ = fixture()
        p = compiled.parameters(); p.update(width=8, depth=2, dropout=0.)
        for kind in compiled.KINDS:
            torch.manual_seed(157)
            model, xs, graphs, _ = reference.setup(a, p, kind, 'cpu')
            torch.nn.init.normal_(model.readout[-1].weight, std=.1)
            replacement, _, _, _ = compiled.setup(a, p, kind, 'cpu')
            replacement.load_state_dict(model.state_dict(), strict=True)
            self.assertEqual(set(model.state_dict()), set(replacement.state_dict()))
            model.eval(); replacement.eval()
            queries = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None])
            expected, actual = model(xs, graphs[0], queries), replacement(xs, graphs[0], queries)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            actual.sum().backward(); expected.sum().backward()
            for x, y in zip(model.parameters(), replacement.parameters()):
                torch.testing.assert_close(x.grad, y.grad, atol=0, rtol=0)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
