"""Exact reuse, matched shared rates, and nontrivial discrete action checks."""
import unittest

import torch

from improve import cooperative_graph_v1 as operators
from improve import cooperative_graph_v3 as old
from improve import cooperative_graph_v4 as revised
from improve.test_nested_anchor_v2 import fixture


class SharedCooperativeChecks(unittest.TestCase):
    def test_full_replays_old_shared_forward_gradients_and_rng(self):
        a, _ = fixture()
        p = dict(revised.parameters(), width=8, depth=2, dropout=.1, evaluation_samples=3)
        torch.manual_seed(551)
        previous, xs, graphs, folds = old.setup(a, p, 'shared_policy', 'cpu')
        current, _, _, _ = revised.setup(a, p, 'full', 'cpu')
        torch.nn.init.normal_(previous.readout[-1].weight, std=.2)
        current.load_state_dict(previous.state_dict())
        query = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None])[:, folds[0]]
        for training in (True, False):
            previous.train(training)
            current.train(training)
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

    def test_control_shares_rates_capacity_and_initial_state_dependence(self):
        a, _ = fixture()
        p = dict(revised.parameters(), width=8, depth=2, dropout=0., evaluation_samples=2)
        torch.manual_seed(84)
        full, xs, graphs, _ = revised.setup(a, p, 'full', 'cpu')
        torch.manual_seed(84)
        soft, _, _, _ = revised.setup(a, p, 'soft_actions', 'cpu')
        for key, value in full.state_dict().items():
            torch.testing.assert_close(value, soft.state_dict()[key], atol=0, rtol=0)
        initial = full.initial(xs, graphs[0])
        changed = initial + torch.randn_like(initial)
        for model in (full, soft):
            logits = model.policy_logits(initial, initial, graphs[0])
            torch.testing.assert_close(logits, logits[:1].expand_as(logits), atol=0, rtol=0)
            torch.testing.assert_close(logits, model.policy_logits(changed, initial, graphs[0]), atol=0, rtol=0)
            gradients = torch.autograd.grad(logits.square().sum(), tuple(model.policy.parameters()), retain_graph=True)
            self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients))
        torch.testing.assert_close(full.policy_logits(initial, initial, graphs[0]),
                                   soft.policy_logits(initial, initial, graphs[0]), atol=0, rtol=0)

    def test_discrete_and_continuous_actions_have_matched_rng_but_distinct_values(self):
        a, _ = fixture()
        p = dict(revised.parameters(), width=8, depth=2, dropout=.1, evaluation_samples=3)
        full, xs, graphs, folds = revised.setup(a, p, 'full', 'cpu')
        soft, _, _, _ = revised.setup(a, p, 'soft_actions', 'cpu')
        torch.nn.init.normal_(full.readout[-1].weight, std=.2)
        soft.load_state_dict(full.state_dict())
        initial = full.initial(xs, graphs[0])
        logits = full.policy_logits(initial, initial, graphs[0])
        noise = full.evaluation_noise[0, 0]
        hard = operators.actions(logits, noise, 1., hard=True)
        continuous = operators.actions(logits, noise, 1., hard=False)
        self.assertTrue(torch.all((hard == 0) | (hard == 1)))
        self.assertTrue(torch.all((continuous > 0) & (continuous < 1)))
        left = torch.autograd.grad(hard[..., 0].sum(), logits, retain_graph=True)[0]
        right = torch.autograd.grad(continuous[..., 0].sum(), logits)[0]
        torch.testing.assert_close(left, right, atol=0, rtol=0)
        query = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None])[:, folds[0]]
        for training in (True, False):
            full.train(training)
            soft.train(training)
            torch.manual_seed(731)
            first = full(xs, graphs[0], query)
            state = torch.get_rng_state().clone()
            torch.manual_seed(731)
            second = soft(xs, graphs[0], query)
            torch.testing.assert_close(state, torch.get_rng_state(), atol=0, rtol=0)
            self.assertGreater(float((first-second).abs().max().detach()), 1e-6)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
