"""Exact candidate reuse and removal of only the declared policy mechanism."""
from copy import deepcopy
import unittest

import torch

from improve import cooperative_graph_v1 as old
from improve import cooperative_graph_v2 as revised
from improve.test_nested_anchor_v2 import fixture


class LocalCooperativeTests(unittest.TestCase):
    def test_full_is_exact_previous_no_lookahead_predictor_and_gradient(self):
        a, _ = fixture()
        p = revised.parameters(); p.update(width=8, depth=2, dropout=.1, evaluation_samples=3)
        torch.manual_seed(551)
        previous, xs, graphs, folds = old.setup(a, p, 'no_lookahead', 'cpu')
        current, _, _, _ = revised.setup(a, p, 'full', 'cpu')
        torch.nn.init.normal_(previous.readout[-1].weight, std=.2)
        current.load_state_dict(previous.state_dict())
        query = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None])[:, folds[0]]
        for training in (True, False):
            previous.train(training); current.train(training)
            torch.manual_seed(524)
            expected = previous(xs, graphs[0], query)
            state = torch.get_rng_state().clone()
            torch.manual_seed(524)
            actual = current(xs, graphs[0], query)
            torch.testing.assert_close(expected, actual, atol=0, rtol=0)
            torch.testing.assert_close(state, torch.get_rng_state(), atol=0, rtol=0)
            left = torch.autograd.grad(expected.sum(), tuple(previous.parameters()))
            right = torch.autograd.grad(actual.sum(), tuple(current.parameters()))
            for x, y in zip(left, right):
                torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_static_policy_ignores_evolving_states_but_retains_trainable_policy(self):
        a, _ = fixture()
        p = revised.parameters(); p.update(width=8, depth=2, dropout=0., evaluation_samples=2)
        full, xs, graphs, _ = revised.setup(a, p, 'full', 'cpu')
        static = deepcopy(full); static.kind = 'static_policy'
        initial = full.initial(xs, graphs[0])
        changed = initial + torch.randn_like(initial)
        torch.testing.assert_close(static.policy_logits(initial, initial, graphs[0]),
                                   static.policy_logits(changed, initial, graphs[0]), atol=0, rtol=0)
        self.assertGreater((full.policy_logits(initial, initial, graphs[0]) -
                            full.policy_logits(changed, initial, graphs[0])).abs().max().item(), 1e-5)
        gradients = torch.autograd.grad(static.policy_logits(changed, initial, graphs[0]).square().sum(),
                                        tuple(static.policy.parameters()))
        self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients))


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
