"""Discrete communication, causal policy inputs and graph/query boundaries."""
from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np
import torch

from improve import cooperative_graph_v1 as cooperative
from improve.test_nested_anchor_v2 import fixture


class CooperativeGraphTests(unittest.TestCase):
    def test_edge_requires_sender_and_receiver_and_keeps_physical_normalizer(self):
        edges = torch.tensor([[0, 1, 2], [2, 2, 0]])
        enabled = torch.tensor([[0, 1], [1, 0], [1, 0]])
        probabilities = torch.stack((enabled, 1 - enabled), dim=-1).double()
        weights = cooperative.edge_weights(probabilities, edges)
        torch.testing.assert_close(weights, torch.tensor([1., 0., 0.], dtype=torch.float64))
        messages = torch.tensor([[2., 6.], [10., 20.], [30., 40.]], dtype=torch.float64)
        degree = torch.tensor([1., 1., 2.], dtype=torch.float64)
        result = cooperative.weighted_mean(messages, edges[1], degree, weights)
        expected = torch.tensor([[0., 0.], [0., 0.], [1., 3.]], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        for send in (0, 1):
            for receive in (0, 1):
                enabled = torch.tensor([[0, send], [receive, 0]])
                p = torch.stack((enabled, 1 - enabled), dim=-1).double()
                self.assertEqual(cooperative.edge_weights(p, torch.tensor([[0], [1]])).item(), send * receive)

    def test_straight_through_gradient_is_the_declared_soft_surrogate(self):
        torch.manual_seed(622)
        logits = torch.randn(9, 2, 2, dtype=torch.float64, requires_grad=True)
        noise = torch.randn_like(logits)
        coeff = torch.randn_like(logits)
        hard = cooperative.actions(logits, noise, .7, True)
        soft = cooperative.actions(logits, noise, .7, False)
        self.assertTrue(torch.all((hard == 0) | (hard == 1)))
        torch.testing.assert_close(hard.sum(-1), torch.ones(9, 2, dtype=torch.float64), atol=0, rtol=0)
        gh = torch.autograd.grad((hard * coeff).sum(), logits, retain_graph=True)[0]
        gs = torch.autograd.grad((soft * coeff).sum(), logits)[0]
        torch.testing.assert_close(gh, gs, atol=0, rtol=0)
        self.assertGreater(gh.abs().sum().item(), 0)

    def test_no_lookahead_removes_neighbor_state_from_policy_only(self):
        torch.manual_seed(68)
        policy = cooperative.Policy(4, 2).double()
        graph = cooperative.Context(edges=torch.tensor([[1, 2], [0, 0]]),
            relation=torch.tensor([0, 1]), counts=torch.zeros(3, 2, dtype=torch.float64),
            degree=torch.tensor([2., 1., 1.], dtype=torch.float64))
        h = torch.randn(3, 4, dtype=torch.float64)
        changed = h.clone(); changed[1] += 5
        torch.testing.assert_close(policy(h, graph, False)[0], policy(changed, graph, False)[0], atol=0, rtol=0)
        self.assertGreater((policy(h, graph, True)[0] - policy(changed, graph, True)[0]).abs().max().item(), 1e-5)
        for lookahead in (True, False):
            gradients = torch.autograd.grad(policy(h, graph, lookahead).square().sum(), tuple(policy.parameters()))
            self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients))

    def test_held_out_queries_never_enter_policy_or_environment_support(self):
        a, _ = fixture()
        ids = np.flatnonzero(a['folds'] == 0)
        expected = cooperative.context(a, ids, 'cpu')
        changed = {k: v.copy() for k, v in a.items()}
        changed['y_train'][ids] = 1 - changed['y_train'][ids]
        changed['pairs_val'] = a['pairs_train'][:, ::-1]
        changed['y_val'] = np.full(len(a['y_train']), 999.)
        changed['anchor_oof'] = np.full(len(a['y_train']), 345.)
        actual = cooperative.context(changed, ids, 'cpu')
        for name in ('edges', 'relation', 'counts', 'degree'):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), atol=0, rtol=0)
        forbidden = set(map(tuple, (a['pairs_train'][:, ids] + a['pair_offset'][:, None]).T))
        forbidden |= {x[::-1] for x in list(forbidden)}
        self.assertFalse(forbidden & set(map(tuple, expected.edges.T.tolist())))
        self.assertTrue(torch.all(expected.edges[0] != expected.edges[1]))

    def test_saved_noise_replay_gradients_and_conditional_reindexing(self):
        torch.manual_seed(1021)
        a, _ = fixture()
        p = cooperative.parameters(); p.update(width=8, depth=2, dropout=0., evaluation_samples=3)
        model, xs, graphs, folds = cooperative.setup(a, p, 'full', 'cpu')
        query = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None])[:, folds[0]]
        model.eval()
        zero = model(xs, graphs[0], query)
        torch.testing.assert_close(zero, torch.zeros_like(zero), atol=0, rtol=0)
        torch.nn.init.normal_(model.readout[-1].weight, std=.1)
        before = torch.get_rng_state().clone()
        expected = model(xs, graphs[0], query)
        torch.testing.assert_close(before, torch.get_rng_state(), atol=0, rtol=0)
        expected.square().sum().backward()
        self.assertGreater(model.policy.trunk[-1].weight.grad.abs().sum().item(), 0)
        replay = model(xs, graphs[0], query.flip(1)).flip(0)
        chunks = torch.cat([model(xs, graphs[0], query[:, i:i + 3]) for i in range(0, query.shape[1], 3)])
        torch.testing.assert_close(expected, replay, atol=3e-7, rtol=3e-7)
        torch.testing.assert_close(expected, chunks, atol=3e-7, rtol=3e-7)
        orders = [torch.randperm(len(x)) for x in xs]
        offsets = np.r_[0, np.cumsum(a['node_counts'])[:-1]]
        permutation = torch.cat([order + int(offset) for order, offset in zip(orders, offsets)])
        inverse = permutation.argsort()
        graph = graphs[0]
        moved = replace(graph, edges=inverse[graph.edges], counts=graph.counts[permutation],
                        degree=graph.degree[permutation])
        alternative = deepcopy(model)
        alternative.evaluation_noise.copy_(model.evaluation_noise[:, :, permutation])
        actual = alternative([x[order] for x, order in zip(xs, orders)], moved, inverse[query])
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-7)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
