import unittest

import numpy as np
import torch

from improve.neural_common_neighbor import neighbor_pool, context, setup


class CommonNeighborTests(unittest.TestCase):
    def test_pool_is_actual_intersection_and_reaches_only_witness_states(self):
        adjacency = torch.zeros((6, 6), dtype=torch.bool)
        edges = torch.tensor([[0, 0, 1, 1], [2, 3, 3, 4]])
        adjacency[edges[0], edges[1]] = True
        adjacency[edges[1], edges[0]] = True
        hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2).requires_grad_()
        typed = torch.ones_like(hidden)
        pairs = torch.tensor([[0, 0], [1, 5]])
        value = neighbor_pool(adjacency, pairs, hidden, typed, 'full')
        torch.testing.assert_close(value[0], hidden[3] + 1)
        torch.testing.assert_close(value[1], torch.zeros(2))
        value.sum().backward()
        self.assertEqual(torch.count_nonzero(hidden.grad.sum(1)).item(), 1)
        self.assertEqual(hidden.grad[3].sum().item(), 2)

    def test_count_null_ignores_neighbor_states_and_query_order(self):
        adjacency = torch.ones((5, 5), dtype=torch.bool)
        adjacency.fill_diagonal_(False)
        hidden, typed = torch.randn(5, 4), torch.randn(5, 4)
        pairs = torch.tensor([[0, 1, 2], [4, 3, 0]])
        first = neighbor_pool(adjacency, pairs, hidden, typed, 'count_only')
        second = neighbor_pool(adjacency, pairs.flip(1), hidden * 100, typed, 'count_only').flip(0)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        torch.testing.assert_close(neighbor_pool(adjacency, pairs, hidden, typed, 'node_only'), torch.zeros_like(first))

    def test_equal_counts_can_have_distinct_neural_witness_information(self):
        adjacency = torch.zeros((5, 5), dtype=torch.bool)
        edges = torch.tensor([[0, 0, 1, 2], [3, 4, 3, 4]])
        adjacency[edges[0], edges[1]] = True
        adjacency[edges[1], edges[0]] = True
        hidden = torch.zeros(5, 2)
        hidden[3, 0] = 1
        hidden[4, 1] = 1
        typed = torch.ones(5, 2)
        pairs = torch.tensor([[0, 0], [1, 2]])
        counts = neighbor_pool(adjacency, pairs, hidden, typed, 'count_only')
        full = neighbor_pool(adjacency, pairs, hidden, typed, 'full')
        torch.testing.assert_close(counts[0], counts[1])
        self.assertFalse(torch.equal(full[0], full[1]))
        permutation = torch.tensor([2, 4, 0, 1, 3])
        inverse = torch.argsort(permutation)
        replay = neighbor_pool(adjacency[permutation][:, permutation], inverse[pairs],
                               hidden[permutation], typed[permutation], 'full')
        torch.testing.assert_close(full, replay, atol=0, rtol=0)

    def test_query_deleted_before_counts_and_pair_support(self):
        a = dict(node_counts=np.array([3, 2]), nd=np.array(2), pair_offset=np.array([0, 3]),
            pairs_train=np.array([[0, 1, 2], [0, 1, 1]]), y_train=np.array([1, 1, 0]),
            folds=np.array([0, 1, 0]), edges=np.array([[0, 1, 0, 1], [3, 4, 1, 2]]),
            relation=np.array([0, 0, 1, 1]), target_codes=np.array([0, 3, -1, -1]),
            x_0=np.ones((3, 2), np.float32), x_1=np.ones((2, 4), np.float32))
        g = context(a, 0, 'cpu')
        self.assertFalse(g[3][0, 3])
        self.assertFalse(g[3][3, 0])
        self.assertEqual(g[2][0, 0].item(), 0)
        self.assertFalse(torch.diag(g[3]).any())
        p = dict(width=8, depth=1, dropout=0., folds=2)
        states = []
        for kind in ('full', 'count_only', 'node_only', 'union'):
            torch.manual_seed(7)
            model, _, _, _ = setup(a, p, kind, 'cpu')
            states.append(model.state_dict())
        for s in states[1:]:
            for name in s:
                torch.testing.assert_close(s[name], states[0][name], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
