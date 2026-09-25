import unittest

import numpy as np
import torch

from improve import overlap_block_models as m


class TestOverlapBlock(unittest.TestCase):
    def test_sparse_softmax_matches_dense_and_gradient(self):
        edge = torch.tensor([[0, 1, 2, 0, 2], [1, 1, 1, 2, 2]])
        x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        y = m.edge_softmax(x, edge[1], 3)
        expected = torch.empty_like(x)
        for d in (1, 2):
            idx = edge[1] == d
            expected[idx] = x[idx].softmax(0)
        torch.testing.assert_close(y, expected)
        weight = torch.randn_like(y)
        a = torch.autograd.grad((weight*y).sum(), x, retain_graph=True)[0]
        b = torch.autograd.grad((weight*expected).sum(), x)[0]
        torch.testing.assert_close(a, b)

    def test_delete_query_before_counts_and_inverse(self):
        a = dict(pairs_train=np.array([[0, 1, 2], [0, 1, 0]]), nd=np.array(2),
                 target_codes=np.array([0, 3, -1]),
                 edges=np.array([[0, 1, 0], [3, 4, 2]]), relation=np.array([0, 0, 1]))
        edges, relation = m.masked_physical(a, [0])
        arcs, kinds, counts = m.graph_inputs(edges, relation, np.empty((2, 0), dtype=int),
                                             5, 2, 'cpu')
        self.assertFalse(((arcs[0] == 0) & (arcs[1] == 3)).any())
        self.assertFalse(((arcs[0] == 3) & (arcs[1] == 0)).any())
        self.assertEqual(float(counts[0, 0]), 0.)
        self.assertGreater(float(counts[1, 0]), 0.)

    def test_stochastic_parameters_receive_gradients(self):
        torch.manual_seed(11)
        model = m.OverlapBlock([5, 4], [4, 3], 1, width=8, depth=1,
                               heads=2, communities=4, dropout=0.)
        graph = m.graph_inputs(np.array([[0, 1, 2], [4, 5, 6]]), np.zeros(3, dtype=int),
                               m.expander_edges(7, 2, 19), 7, 1, 'cpu')
        xs = [torch.randn(4, 5), torch.randn(3, 4)]
        pairs = torch.tensor([[0, 2, 3], [5, 4, 6]])
        pred, kl = model(xs, graph, pairs, torch.Generator().manual_seed(8))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, torch.tensor([1., 0., 1.])) + .01*kl
        loss.backward()
        for name in ('stick_a', 'stick_b', 'membership.weight', 'strength.weight',
                     'log_std.weight', 'layers.0.qkv.weight'):
            grad = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(float(grad.abs().sum()), 0., name)

    def test_posterior_scores_query_order_and_chunk_invariance(self):
        torch.manual_seed(31)
        model = m.OverlapBlock([5, 4], [4, 3], 1, width=8, depth=1,
                               heads=2, communities=4, dropout=0.)
        graph = m.graph_inputs(np.array([[0, 1, 2], [4, 5, 6]]), np.zeros(3, dtype=int),
                               m.expander_edges(7, 2, 19), 7, 1, 'cpu')
        xs = [torch.randn(4, 5), torch.randn(3, 4)]
        pairs = torch.tensor([[0, 2, 3], [5, 4, 6]])
        full = m.posterior_prediction(model, xs, [graph], pairs, samples=9)
        reverse = m.posterior_prediction(model, xs, [graph], pairs.flip(1), samples=9).flip(0)
        chunked = torch.cat([m.posterior_prediction(model, xs, [graph], q, samples=9)
                             for q in torch.tensor_split(pairs, 2, dim=1)])
        torch.testing.assert_close(full, reverse, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(full, chunked, atol=1e-6, rtol=1e-5)

    def test_binary_concrete_correct_zero_temperature_frequency(self):
        logits = torch.full((40000,), torch.logit(torch.tensor(.2)))
        values = m.concrete_sample(logits, .01, torch.Generator().manual_seed(55))
        self.assertAlmostEqual(float(values.mean()), .2, delta=.01)


if __name__ == '__main__':
    unittest.main()
