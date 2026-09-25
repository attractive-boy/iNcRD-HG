"""Mathematical decomposition, gradient, mask and query-isolation checks."""
import unittest
import numpy as np
import torch
from torch.nn import functional as F

from improve import neighbor_type_contrastive_v1 as core
from improve.test_nested_anchor_v2 import fixture


class ContrastiveTests(unittest.TestCase):
    def test_source_decomposition_and_relabeling(self):
        torch.manual_seed(72)
        a, _ = fixture()
        p = dict(core.parameters(), width=8, dropout=0.)
        model, xs, graphs, _ = core.setup(a, p, 'full', 'cpu')
        adjacency = graphs[0]['adjacency'].double()
        initial = torch.randn(14, 8, dtype=torch.float64, requires_grad=True)
        actual = core.source_propagation(initial, adjacency, 8, 3)
        dense = adjacency.to_dense()
        operator = sum(torch.linalg.matrix_power(dense, k) for k in range(4)) / 4
        expected = torch.stack((operator[:, :8] @ initial[:8], operator[:, 8:] @ initial[8:]), 1)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(actual.sum(1), operator @ initial, atol=1e-12, rtol=1e-12)
        permutation = torch.cat((torch.randperm(8), 8+torch.randperm(6)))
        permuted = core.source_propagation(initial[permutation], dense[permutation][:, permutation].to_sparse(), 8, 3)
        torch.testing.assert_close(permuted, actual[permutation], atol=1e-12, rtol=1e-12)

    def test_type_weighted_logits_and_standard_ssm_reduction(self):
        torch.manual_seed(15)
        left = torch.randn(5, 2, 4, dtype=torch.float64, requires_grad=True)
        right = torch.randn(5, 2, 4, dtype=torch.float64, requires_grad=True)
        denominator = left.sum(1).norm(dim=-1)[:, None]*right.sum(1).norm(dim=-1)[None, :] + 1e-8
        terms = [left[:, i] @ right[:, j].T / denominator for i in range(2) for j in range(2)]
        ordinary = sum(terms)
        expected = 1.2*(terms[0]+terms[2]) + .8*(terms[1]+terms[3])
        diagonal = torch.eye(5, dtype=torch.bool)
        expected = torch.where(diagonal, ordinary, expected)
        actual = core.type_logits(left, right, (1.2, .8))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        plain = core.type_logits(left, right, (1., 1.))
        torch.testing.assert_close(plain, ordinary, atol=1e-12, rtol=1e-12)
        allowed = ~diagonal
        loss = core.directional_loss(left, right, (1., 1.), .2, allowed)
        torch.testing.assert_close(loss, F.cross_entropy(ordinary/.2, torch.arange(5)), atol=1e-12, rtol=1e-12)
        self.assertTrue(torch.autograd.gradcheck(lambda x, y: core.type_logits(x, y, (1.2, .8)), (left, right)))

    def test_repeated_endpoints_have_no_false_negative_and_controls_are_matched(self):
        torch.manual_seed(32)
        rna = torch.randn(3, 2, 4, requires_grad=True)
        drug = torch.randn(3, 2, 4, requires_grad=True)
        p = core.parameters()
        repeated = torch.tensor([[0, 0, 0], [1, 2, 3]])
        loss = core.contrastive_loss(rna, drug, repeated, 'full', p)
        self.assertEqual(float(loss), 0.)
        a, _ = fixture()
        initial, after = [], []
        for kind in core.KINDS:
            torch.manual_seed(56)
            model, xs, graphs, folds = core.setup(a, dict(p, width=8), kind, 'cpu')
            initial.append(model.state_dict())
            after.append(torch.get_rng_state())
        for other, rng in zip(initial[1:], after[1:]):
            self.assertEqual(initial[0].keys(), other.keys())
            for name in initial[0]:
                torch.testing.assert_close(initial[0][name], other[name], atol=0, rtol=0)
            torch.testing.assert_close(after[0], rng, atol=0, rtol=0)

    def test_query_labels_cannot_change_support(self):
        a, _ = fixture()
        p = dict(core.parameters(), width=8)
        _, _, graphs, folds = core.setup(a, p, 'full', 'cpu')
        for f in range(5):
            ids = folds[f].numpy()
            changed = {k: v.copy() for k, v in a.items()}
            changed['y_train'][ids] = 1-changed['y_train'][ids]
            _, _, changed_graphs, _ = core.setup(changed, p, 'full', 'cpu')
            torch.testing.assert_close(graphs[f]['adjacency'].to_dense(), changed_graphs[f]['adjacency'].to_dense(), atol=0, rtol=0)
            torch.testing.assert_close(graphs[f]['global_edges'], changed_graphs[f]['global_edges'], atol=0, rtol=0)
            edges = set(map(tuple, graphs[f]['global_edges'].numpy().T))
            for pair in (a['pairs_train'][:, ids]+a['pair_offset'][:, None]).T:
                self.assertNotIn(tuple(pair), edges)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
