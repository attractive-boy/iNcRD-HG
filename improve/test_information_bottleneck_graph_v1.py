"""Information objective, undirected mask semantics, and query boundaries."""
import unittest

import numpy as np
import torch

from improve import information_bottleneck_graph_v1 as core


class BottleneckTests(unittest.TestCase):
    def test_isolated_query_has_zero_edge_kl_and_a_finite_prediction(self):
        context = core.BottleneckContext(np.array([[0], [2]]), [2, 2], .1)
        model = core.BottleneckGraph([3, 3], core.parameters(), 'full').eval()
        value, kl, detail = model.forward_with_kl([torch.ones(2, 3), torch.ones(2, 3)],
                                                 context, torch.tensor([[1], [3]]))
        self.assertTrue(torch.isfinite(value).all())
        torch.testing.assert_close(kl, torch.zeros_like(kl), atol=0, rtol=0)
        self.assertEqual(detail['undirected_edges'], 0)

    def test_resource_allocation_completes_only_same_type_and_keeps_ties(self):
        edges = np.array([[0, 1, 2, 3], [4, 4, 5, 5]])
        augmented, origin, audit = core.complete_same_type(edges, [4, 2], .25)
        self.assertEqual(audit['nominal_addition_budget'], 1)
        self.assertEqual(audit['virtual_undirected_edges'], 2)
        self.assertGreater(origin[0, 1], 0)
        self.assertGreater(origin[2, 3], 0)
        self.assertEqual(origin[:4, 4:].nnz, 0)
        np.testing.assert_array_equal(augmented, np.array([[0, 0, 1, 2, 2, 3], [1, 4, 4, 3, 5, 5]]))

    def test_bernoulli_kl_matches_distribution_and_sums_by_graph(self):
        logits = torch.tensor([-2., -.3, .8, 3.], dtype=torch.float64, requires_grad=True)
        prior = torch.tensor([.8, .5, .8, .5], dtype=torch.float64)
        actual = core.bernoulli_kl(logits, prior)
        expected = torch.distributions.kl_divergence(torch.distributions.Bernoulli(logits=logits),
                                                     torch.distributions.Bernoulli(probs=prior))
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
        zero = core.bernoulli_kl(torch.logit(prior), prior)
        torch.testing.assert_close(zero, torch.zeros_like(zero), atol=1e-14, rtol=0)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_reverse_messages_share_one_mask_and_query_fact_is_absent(self):
        edges = np.array([[0, 1, 2, 3], [4, 4, 5, 5]])
        context = core.BottleneckContext(edges, [4, 2], .25)
        pairs = torch.tensor([[0, 2], [5, 4]])
        packed = context.batch(pairs)
        for index, (u, v) in enumerate(packed['edges'].T.tolist()):
            if u == v:
                self.assertEqual(packed['groups'][index], 0)
            else:
                reverse = torch.nonzero((packed['edges'][0] == v) & (packed['edges'][1] == u)).item()
                self.assertEqual(packed['groups'][index], packed['groups'][reverse])
        with self.assertRaises(ValueError):
            context.batch(torch.tensor([[0], [4]]))

    def test_private_noise_and_eval_batch_invariance(self):
        p = core.parameters()
        p.update(width=8, dropout=0.)
        context = core.BottleneckContext(np.array([[0, 1, 2, 3], [4, 4, 5, 5]]), [4, 2], .25)
        xs = [torch.tensor([[1., .2], [.3, 1.], [.6, .4], [.7, .8]]), torch.tensor([[1., .7], [.4, 1.]])]
        pairs = torch.tensor([[0, 2, 3], [5, 4, 4]])
        model = core.BottleneckGraph([2, 2], p, 'full')
        torch.nn.init.normal_(model.readout[-1].weight, std=.02)
        state = torch.random.get_rng_state().clone()
        x = torch.zeros(9)
        one = core.relaxed_bernoulli(x, .67, torch.Generator().manual_seed(6))
        two = core.relaxed_bernoulli(x, .67, torch.Generator().manual_seed(6))
        torch.testing.assert_close(one, two, atol=0, rtol=0)
        torch.testing.assert_close(state, torch.random.get_rng_state(), atol=0, rtol=0)
        model.train()
        value, kl, detail = model.forward_with_kl(xs, context, pairs, torch.Generator().manual_seed(7))
        (value.square().mean() + .001 * kl.mean()).backward()
        self.assertGreater(sum(float(w.grad.abs().sum()) for w in model.key.parameters()), 0.)
        self.assertGreater(detail['virtual_edges'], 0)
        model.eval()
        full, kl_full, _ = model.forward_with_kl(xs, context, pairs)
        chunks = [model.forward_with_kl(xs, context, pairs[:, i:i + 1]) for i in range(3)]
        torch.testing.assert_close(full, torch.cat([v[0] for v in chunks]), atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(kl_full, torch.cat([v[1] for v in chunks]), atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(full, model(xs, context, pairs.flip(1)).flip(0), atol=1e-7, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
