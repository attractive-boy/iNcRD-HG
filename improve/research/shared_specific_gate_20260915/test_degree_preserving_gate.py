"""Smoke/contract tests for the degree-preserving shared/specific gate.

Run with the repo's PyG environment:
    /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -m unittest \
        improve.research.shared_specific_gate_20260915.test_degree_preserving_gate -v

These are contract tests on synthetic tensors, not a performance claim.
"""
import unittest

import torch

from improve.research.shared_specific_gate_20260915.degree_preserving_gate import (
    SharedSpecificGate, degree_preserving_weights)


class TestDegreePreservingWeights(unittest.TestCase):
    def test_per_target_sum_equals_in_degree(self):
        # targets 0,1,2 receive 1,2,3 edges respectively
        dst = torch.tensor([0, 1, 1, 2, 2, 2])
        src = torch.arange(6)
        edge_index = torch.stack([src, dst])
        scores = torch.rand(6) + 0.1
        w = degree_preserving_weights(scores, edge_index, num_nodes=3)
        per_target = torch.zeros(3).index_add_(0, dst, w)
        expected = torch.tensor([1.0, 2.0, 3.0])
        self.assertTrue(torch.allclose(per_target, expected, atol=1e-5),
                        f'{per_target} != {expected}')

    def test_weights_are_nonnegative_and_positive_when_score_positive(self):
        edge_index = torch.tensor([[0, 0, 1, 2], [1, 1, 2, 2]])
        scores = torch.tensor([0.2, 0.8, 0.5, 0.3])
        w = degree_preserving_weights(scores, edge_index, num_nodes=3)
        self.assertTrue((w >= 0).all())
        self.assertTrue((w > 0).all())

    def test_two_node_counts_accepted(self):
        edge_index = torch.tensor([[0, 2], [1, 1]])
        scores = torch.ones(2)
        w = degree_preserving_weights(scores, edge_index, num_nodes=(3, 3))
        per_target = torch.zeros(3).index_add_(0, edge_index[1], w)
        self.assertTrue(torch.allclose(per_target[1], torch.tensor(2.0)))


class TestSharedSpecificGate(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.hidden = 8
        self.n = 5

    def test_zero_shared_is_identity(self):
        gate = SharedSpecificGate(self.hidden)
        h_specific = torch.randn(self.n, self.hidden)
        out = gate(h_specific, torch.zeros_like(h_specific))
        self.assertTrue(torch.allclose(out, h_specific, atol=1e-6))

    def test_gate_starts_nearly_closed(self):
        gate = SharedSpecificGate(self.hidden, init_bias=-2.0)
        h_specific = torch.randn(self.n, self.hidden)
        h_shared = torch.randn(self.n, self.hidden)
        out = gate(h_specific, h_shared)
        delta = (out - h_specific).abs().max().item()
        self.assertLess(delta, 0.3 * h_shared.abs().max().item())

    def test_gate_is_bounded_and_gradients_flow(self):
        gate = SharedSpecificGate(self.hidden, evidence_dim=3)
        h_specific = torch.randn(self.n, self.hidden, requires_grad=True)
        h_shared = torch.randn(self.n, self.hidden, requires_grad=True)
        evidence = torch.randn(self.n, 3)
        out = gate(h_specific, h_shared, evidence)
        self.assertTrue(((gate.last_gate >= 0) & (gate.last_gate <= 1)).all())
        out.sum().backward()
        self.assertIsNotNone(h_shared.grad)
        self.assertGreater(h_shared.grad.abs().sum().item(), 0.0)
        self.assertGreater(gate.gate.weight.grad.abs().sum().item(), 0.0)

    def test_opening_the_gate_moves_towards_shared(self):
        gate = SharedSpecificGate(self.hidden, init_bias=2.0)
        h_specific = torch.zeros(self.n, self.hidden)
        h_shared = torch.ones(self.n, self.hidden)
        out = gate(h_specific, h_shared)
        self.assertTrue((out > 0.5).all())


if __name__ == '__main__':
    unittest.main()
