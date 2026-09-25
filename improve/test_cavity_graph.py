import unittest

import numpy as np
import torch

from improve.cavity_graph import cavity_mean, support_graph


def arcs(edges):
    x = torch.as_tensor(edges, dtype=torch.long)
    n = x.shape[1]
    return torch.cat((x, x.flip(0)), dim=1), torch.cat((torch.arange(n, 2 * n), torch.arange(n)))


class CavityGraphTest(unittest.TestCase):
    def test_incidence_subtraction_matches_explicit_line_graph_and_gradients(self):
        edges, inverse = arcs([[0, 0, 1, 2], [1, 2, 2, 3]])
        torch.manual_seed(81)
        x = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
        expected = []
        for i in range(8):
            ids = torch.where((edges[1] == edges[0, i]) & (torch.arange(8) != inverse[i]))[0]
            expected.append(x[ids].mean(0) if len(ids) else x[inverse[i]])
        expected = torch.stack(expected)
        actual = cavity_mean(x, edges, inverse, 4)
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
        weight = torch.randn_like(actual)
        ga = torch.autograd.grad((actual * weight).sum(), x)[0]
        gb = torch.autograd.grad((expected * weight).sum(), x)[0]
        torch.testing.assert_close(ga, gb, atol=1e-14, rtol=1e-14)

    def test_inverse_edge_has_zero_jacobian_away_from_leaf(self):
        edges, inverse = arcs([[0, 0], [1, 2]])
        x = torch.randn(4, 1, requires_grad=True)
        g = torch.autograd.grad(cavity_mean(x, edges, inverse, 3)[0].sum(), x)[0]
        self.assertEqual(float(g[inverse[0]]), 0.)
        self.assertEqual(float(g[inverse[1]]), 1.)
        g = torch.autograd.grad(cavity_mean(x, edges, inverse, 3, backtracking=True)[0].sum(), x)[0]
        self.assertEqual(float(g[inverse[0]]), .5)

    def test_leaf_return_is_explicit_and_finite(self):
        edges, inverse = arcs([[0], [1]])
        x = torch.tensor([[2.], [3.]])
        torch.testing.assert_close(cavity_mean(x, edges, inverse, 2), x.flip(0))
        torch.testing.assert_close(cavity_mean(x, edges, inverse, 2, leaf_return=False), torch.zeros_like(x))

    def test_parallel_relations_cannot_reintroduce_the_previous_node(self):
        edges, inverse = arcs([[0, 0, 0], [1, 1, 2]])
        x = torch.tensor([[1.], [2.], [3.], [4.], [8.], [16.]], requires_grad=True)
        actual = cavity_mean(x, edges, inverse, 3)
        self.assertEqual(float(actual[0]), 16.)
        self.assertEqual(float(actual[1]), 16.)
        self.assertEqual(float(actual[2]), 6.)
        gradient = torch.autograd.grad(actual[0].sum(), x)[0]
        self.assertEqual(float(gradient[3] + gradient[4]), 0.)

    def test_query_positive_and_zero_are_removed_before_counts(self):
        a = dict(edges=np.array([[0, 1], [2, 3]]), relation=np.array([0, 0]),
                 target_codes=np.array([0, 3]), node_counts=np.array([2, 2]),
                 pairs_train=np.array([[0, 1, 0, 1], [0, 1, 1, 0]]),
                 y_train=np.array([1, 1, 0, 0]), nd=np.array(2), pair_offset=np.array([0, 2]))
        edges, relation, reverse, counts, _ = support_graph(a, np.array([0, 2]), 'full', 'cpu')
        self.assertEqual(int((edges[0] == 0).sum()), 0)
        self.assertEqual(float(counts[0].sum()), 0.)
        torch.testing.assert_close(edges[:, reverse], edges.flip(0))
        _, _, _, positive_counts, _ = support_graph(a, np.array([0, 2]), 'positive_only', 'cpu')
        torch.testing.assert_close(counts, positive_counts)


if __name__ == '__main__':
    unittest.main()
