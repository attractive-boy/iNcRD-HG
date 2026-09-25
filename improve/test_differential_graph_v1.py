"""Dense references, cancellation and matched-parameter graph symmetries."""
import math
import unittest

import torch

from improve import differential_graph_v1 as core


class DifferentialChecks(unittest.TestCase):
    def test_sparse_matches_dense_values_and_all_gradients(self):
        torch.manual_seed(41)
        q = torch.randn(5, 2, 2, 3, dtype=torch.float64, requires_grad=True)
        k, v = [torch.randn(5, 2, 3, dtype=torch.float64, requires_grad=True) for _ in range(2)]
        edges = torch.tensor([[0, 1, 2, 3, 4, 0, 2, 1], [1, 2, 3, 4, 0, 3, 0, 4]])
        e = torch.randn(8, 2, 3, dtype=torch.float64, requires_grad=True)
        lam = torch.rand(5, 2, dtype=torch.float64, requires_grad=True)
        for sign in (-1, 1):
            actual = core.paired_aggregate(q, k, v, edges, e, lam, sign)
            rows = []
            for dst in range(5):
                ids = torch.where(edges[1] == dst)[0]
                src = edges[0, ids]
                per_head = []
                for head in range(2):
                    # Explicit per-node paired queries, sharing exactly one K/V head.
                    score1 = ((k[src, head] + e[ids, head]) * q[dst, head, 0]).sum(-1) / math.sqrt(3)
                    score2 = ((k[src, head] + e[ids, head]) * q[dst, head, 1]).sum(-1) / math.sqrt(3)
                    weights = (score1.softmax(0) + sign * lam[dst, head] * score2.softmax(0)) / (1 + lam[dst, head])
                    per_head.append((weights[:, None] * v[src, head]).sum(0))
                rows.append(torch.stack(per_head))
            expected = torch.stack(rows)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            variables = (q, k, v, e, lam)
            ga = torch.autograd.grad(actual.square().sum(), variables, retain_graph=True)
            gb = torch.autograd.grad(expected.square().sum(), variables, retain_graph=True)
            for a, b in zip(ga, gb):
                torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)

    def test_common_mode_cancellation_and_l1_bound(self):
        torch.manual_seed(72)
        q = torch.randn(4, 2, 2, 3, dtype=torch.float64)
        q[:, :, 1] = q[:, :, 0]
        k = torch.randn(4, 2, 3, dtype=torch.float64)
        v = torch.randn(4, 2, 3, dtype=torch.float64)
        edges = torch.stack((torch.arange(4).repeat(4), torch.arange(4).repeat_interleave(4)))
        e = torch.zeros(16, 2, 3, dtype=torch.float64)
        lam = torch.ones(4, 2, dtype=torch.float64)
        actual = core.paired_aggregate(q, k, v, edges, e, lam, -1)
        torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)
        q[:, :, 1] *= -2
        lam.fill_(.7)
        weights = core.paired_weights(q, k, edges, e, lam, -1)
        for dst in range(4):
            w = weights[edges[1] == dst]
            self.assertTrue((w.abs().sum(0) <= 1 + 1e-14).all())
            torch.testing.assert_close(w.sum(0), torch.full((2,), .3 / 1.7, dtype=torch.float64), atol=1e-14, rtol=1e-14)

    def test_matched_parameters_and_node_query_equivariance(self):
        p = dict(core.parameters(), width=8, heads=2, depth=1, dropout=0.)
        torch.manual_seed(42)
        full = core.DifferentialGraph([3], 2, p, 'full').double().eval()
        full_rng = torch.get_rng_state()
        torch.manual_seed(42)
        control = core.DifferentialGraph([3], 2, p, 'additive_attention').double().eval()
        torch.testing.assert_close(full_rng, torch.get_rng_state(), atol=0, rtol=0)
        for name, value in full.state_dict().items():
            torch.testing.assert_close(value, control.state_dict()[name], atol=0, rtol=0)
        torch.nn.init.normal_(full.readout[-1].weight, std=.1)
        xs = [torch.randn(5, 3, dtype=torch.float64)]
        edges = torch.tensor([[0, 1, 2, 3, 4, 0, 2], [1, 2, 3, 4, 0, 2, 0]])
        relation, counts = torch.arange(7) % 4, torch.rand(5, 2, dtype=torch.float64)
        pairs = torch.tensor([[0, 2, 3], [3, 4, 1]])
        permutation = torch.tensor([3, 1, 4, 0, 2])
        inverse = permutation.argsort()
        original = full(xs, (edges, relation, counts), pairs)
        renamed = full([xs[0][permutation]], (inverse[edges], relation, counts[permutation]), inverse[pairs])
        torch.testing.assert_close(original, renamed, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(original, full(xs, (edges, relation, counts), pairs.flip(1)).flip(0), atol=0, rtol=0)
        original.square().sum().backward()
        for name, value in full.named_parameters():
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(torch.isfinite(value.grad).all(), name)
        self.assertGreater(full.layers[0].q.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(full.layers[0].gate.weight.grad.abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
