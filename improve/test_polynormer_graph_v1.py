"""Independent dense references and symmetry checks for the adaptation."""
import unittest

import torch

from improve import polynormer_graph_v1 as core


class PolynormerChecks(unittest.TestCase):
    def test_linear_attention_matches_dense_values_and_gradients(self):
        torch.manual_seed(31)
        q, k, v = [torch.rand(7, 2, 3, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        actual = core.linear_attention(q, k, v)
        weights = torch.einsum('ihd,jhd->hij', q, k)
        weights = weights / weights.sum(-1, keepdim=True)
        expected = torch.einsum('hij,jhe->ihe', weights, v)
        torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
        g1 = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
        g2 = torch.autograd.grad(expected.square().sum(), (q, k, v))
        for a, b in zip(g1, g2):
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)

    def test_local_aggregate_matches_explicit_neighborhoods(self):
        torch.manual_seed(9)
        layer = core.LocalAttention(8, 2, 4).double()
        x = torch.randn(5, 8, dtype=torch.float64)
        edges = torch.tensor([[0, 2, 3, 1, 4, 0], [1, 1, 2, 2, 3, 3]])
        relation = torch.tensor([0, 1, 2, 3, 1, 0])
        actual = layer(x, edges, relation)
        values = layer.value(x).reshape(5, 2, 4)
        expected = torch.zeros_like(values)
        for dst in range(5):
            ids = torch.where(edges[1] == dst)[0]
            if not len(ids):
                continue
            src = edges[0, ids]
            score = ((values[src] * layer.att_source).sum(-1)
                     + (values[dst] * layer.att_target).sum(-1)
                     + layer.relation_bias(relation[ids]))
            weight = torch.nn.functional.leaky_relu(score, .2).softmax(0)
            expected[dst] = (weight[..., None] * values[src]).sum(0)
        torch.testing.assert_close(actual, expected.flatten(1), atol=1e-13, rtol=1e-13)

    def test_paired_parameters_and_node_query_equivariance(self):
        p = dict(core.parameters(), width=8, heads=2, local_depth=2, global_depth=1, dropout=0.)
        torch.manual_seed(51)
        full = core.PolynomialGraph([3], 2, p, 'full').double().eval()
        torch.manual_seed(51)
        control = core.PolynomialGraph([3], 2, p, 'additive_update').double().eval()
        for (ka, a), (kb, b) in zip(full.state_dict().items(), control.state_dict().items()):
            self.assertEqual(ka, kb)
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.nn.init.normal_(full.readout[-1].weight, std=.1)
        xs = [torch.randn(6, 3, dtype=torch.float64)]
        edges = torch.tensor([[0, 1, 2, 3, 4, 5, 2, 0], [1, 2, 3, 4, 5, 0, 0, 3]])
        types = torch.arange(8) % 4
        counts = torch.rand(6, 2, dtype=torch.float64)
        pairs = torch.tensor([[0, 2, 4, 1], [3, 5, 1, 0]])
        permutation = torch.tensor([3, 1, 5, 0, 4, 2])
        inverse = permutation.argsort()
        original = full(xs, (edges, types, counts), pairs)
        renamed = full([xs[0][permutation]], (inverse[edges], types, counts[permutation]), inverse[pairs])
        torch.testing.assert_close(original, renamed, atol=1e-12, rtol=1e-12)
        reordered = full(xs, (edges, types, counts), pairs.flip(1)).flip(0)
        torch.testing.assert_close(original, reordered, atol=0, rtol=0)
        original.sum().backward()
        for name, value in full.named_parameters():
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(torch.isfinite(value.grad).all(), name)


if __name__ == '__main__':
    unittest.main()
