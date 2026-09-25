import unittest

import numpy as np
import torch

from improve import neural_common_neighbor as old
from improve.neural_neighborhood_union import union_pool, typed_permutation, setup


class UnionTests(unittest.TestCase):
    def test_actual_union_and_count_null_use_identical_support(self):
        adjacency = torch.zeros((6, 6), dtype=torch.bool)
        edges = torch.tensor([[0, 0, 1, 1], [2, 3, 3, 4]])
        adjacency[edges[0], edges[1]] = True
        adjacency[edges[1], edges[0]] = True
        hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2).requires_grad_()
        typed = torch.ones_like(hidden)
        pairs = torch.tensor([[0], [1]])
        identity = torch.arange(6)
        full = union_pool(adjacency, pairs, hidden, typed, 'full', identity)
        count = union_pool(adjacency, pairs, hidden, typed, 'count_only', identity)
        torch.testing.assert_close(full[0], (hidden[2:5] + 1).sum(0) / 3**.5)
        torch.testing.assert_close(count[0], torch.full((2,), 3**.5))
        full.sum().backward()
        self.assertEqual(torch.nonzero(hidden.grad.sum(1)).flatten().tolist(), [2, 3, 4])
        torch.testing.assert_close(count, union_pool(adjacency, pairs, hidden * 100, typed,
                                                     'count_only', identity), atol=0, rtol=0)

    def test_shuffle_preserves_types_and_alignment_is_query_equivariant(self):
        permutation = torch.as_tensor(typed_permutation([3, 4], 5))
        self.assertEqual(set(permutation[:3].tolist()), {0, 1, 2})
        self.assertEqual(set(permutation[3:].tolist()), {3, 4, 5, 6})
        adjacency = torch.rand(7, 7) > .6
        hidden, typed = torch.randn(7, 3), torch.randn(7, 3)
        pairs = torch.tensor([[0, 1, 2], [3, 5, 6]])
        value = union_pool(adjacency, pairs, hidden, typed, 'shuffled', permutation)
        replay = union_pool(adjacency, pairs.flip(1), hidden, typed, 'shuffled', permutation).flip(0)
        torch.testing.assert_close(value, replay)
        expected = union_pool(adjacency, pairs, hidden[permutation], typed, 'full', torch.arange(7))
        torch.testing.assert_close(value, expected, atol=0, rtol=0)

    def test_reused_arms_have_exact_states_outputs_and_deleted_queries(self):
        a = dict(node_counts=np.array([3, 2]), nd=np.array(2), pair_offset=np.array([0, 3]),
            pairs_train=np.array([[0, 1, 2], [0, 1, 1]]), y_train=np.array([1, 1, 0]),
            folds=np.array([0, 1, 0]), edges=np.array([[0, 1, 0, 1], [3, 4, 1, 2]]),
            relation=np.array([0, 0, 1, 1]), target_codes=np.array([0, 3, -1, -1]),
            x_0=np.ones((3, 2), np.float32), x_1=np.ones((2, 4), np.float32))
        p = dict(width=8, depth=1, dropout=0., folds=2, permutation_seed=17)
        pairs = torch.tensor([[0, 1], [3, 4]])
        for new_kind, old_kind in [('full', 'union'), ('node_only', 'node_only')]:
            torch.manual_seed(7)
            prior, px, pg, _ = old.setup(a, p, old_kind, 'cpu')
            torch.manual_seed(7)
            current, cx, cg, _ = setup(a, p, new_kind, 'cpu')
            for key, value in prior.state_dict().items():
                torch.testing.assert_close(value, current.state_dict()[key], atol=0, rtol=0)
            self.assertFalse(cg[0][3][0, 3])
            self.assertFalse(cg[0][3][3, 0])
            self.assertEqual(cg[0][2][0, 0].item(), 0.)
            torch.testing.assert_close(prior(px, pg[0], pairs), current(cx, cg[0], pairs), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
