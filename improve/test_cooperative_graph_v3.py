"""Exact static-policy adoption and newly matched continuous-action control."""
import unittest

import torch

from improve import cooperative_graph_v2 as old
from improve import cooperative_graph_v3 as revised
from improve.test_nested_anchor_v2 import fixture


class StaticCooperativeChecks(unittest.TestCase):
    def test_full_exactly_replays_old_static_forward_gradients_and_rng(self):
        a, _ = fixture()
        p = dict(revised.parameters(), width=8, depth=2, dropout=.1, evaluation_samples=3)
        torch.manual_seed(551)
        previous, xs, graphs, folds = old.setup(a, p, 'static_policy', 'cpu')
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

    def test_new_control_shares_initial_policy_and_all_parameters(self):
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
            torch.testing.assert_close(model.policy_logits(initial, initial, graphs[0]),
                model.policy_logits(changed, initial, graphs[0]), atol=0, rtol=0)
        torch.testing.assert_close(full.policy_logits(changed, initial, graphs[0]),
                                   soft.policy_logits(changed, initial, graphs[0]), atol=0, rtol=0)

    def test_shared_policy_removes_node_specific_rates_but_uses_all_policy_parameters(self):
        a, _ = fixture()
        p = dict(revised.parameters(), width=8, depth=2, dropout=0., evaluation_samples=2)
        model, xs, graphs, _ = revised.setup(a, p, 'shared_policy', 'cpu')
        initial = model.initial(xs, graphs[0])
        logits = model.policy_logits(initial, initial, graphs[0])
        torch.testing.assert_close(logits, logits[:1].expand_as(logits), atol=0, rtol=0)
        gradients = torch.autograd.grad(logits.square().sum(), tuple(model.policy.parameters()))
        self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients))


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
