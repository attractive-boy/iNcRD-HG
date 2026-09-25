"""CVaR equivalence, independent blocks, scalar updates, and stage averaging."""
import unittest

import numpy as np
import torch

from improve import staco_ranking_v1 as risk


class RankingMathChecks(unittest.TestCase):
    def test_two_way_cvar_matches_sorted_pair_risk_and_fractional_tails(self):
        positive = torch.tensor([.7, .2, .8, .5], dtype=torch.float64)
        negative = torch.tensor([.8, .1, .4, .6, .3, .9], dtype=torch.float64)
        losses = risk.pair_loss(positive, negative, .5)
        inner = risk.cvar(losses, risk.tail_threshold(losses, .5), .5)
        actual = risk.cvar(inner, risk.tail_threshold(inner, .5), .5)
        expected = risk.pair_loss(positive.sort().values[:2], negative.sort(descending=True).values[:3], .5).mean()
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=0)
        values = torch.tensor([.1, .2, .8, .6, .4], dtype=torch.float64)
        threshold = risk.tail_threshold(values, .5)
        actual = risk.cvar(values, threshold, .5)
        torch.testing.assert_close(actual, torch.tensor((.8+.6+.5*.4)/2.5, dtype=torch.float64), atol=1e-14, rtol=0)
        self.assertEqual(float(threshold), .4)

    def test_scalar_primal_dual_step_and_unselected_coordinates(self):
        p = dict(risk.parameters(), folds=1, proximal_gamma=2.)
        labels = np.array([1, 1, 0, 0, 0, 0])
        state = risk.TailState(labels, np.zeros(6, dtype=int), np.zeros(6), p, 1, 'cpu')
        model = torch.nn.Linear(1, 1, bias=False)
        state.s[0] = .1
        state.begin_stage(model)
        state.s[0] = .2
        state.y[0], state.y[1] = .4, .7
        state.prime = torch.tensor(.25, dtype=torch.float64)
        old_s, old_y = state.s.clone(), state.y.clone()
        probability = torch.tensor([.6, .8, .7, .1, .2, .9], dtype=torch.float64, requires_grad=True)
        batch = risk.RiskBatch(torch.tensor([0]), torch.tensor([0]), torch.tensor([2, 3]), torch.tensor([4, 5]))
        objective, _ = state.block_objective(probability, batch, 'full')
        self.assertAlmostEqual(float(state.y[0]), .51, places=13)
        self.assertAlmostEqual(float(state.s[0]), .19, places=13)
        self.assertAlmostEqual(float(state.prime), .2502, places=13)
        self.assertAlmostEqual(float(objective.detach()), .6478, places=13)
        torch.testing.assert_close(state.s[1:], old_s[1:], atol=0, rtol=0)
        torch.testing.assert_close(state.y[1:], old_y[1:], atol=0, rtol=0)
        gradient = torch.autograd.grad(objective, probability)[0]
        torch.testing.assert_close(gradient, torch.tensor([-1.632, 0, 0, 0, 0, 1.632], dtype=torch.float64), atol=1e-13, rtol=0)

    def test_two_negative_draws_are_separate_and_do_not_consume_model_rng(self):
        labels = np.r_[np.ones(10), np.zeros(40)]
        p = dict(risk.parameters(), folds=1, positive_batch=4, negative_batch=7)
        left = risk.TailState(labels, np.zeros(50, dtype=int), np.zeros(50), p, 845, 'cpu')
        right = risk.TailState(labels, np.zeros(50, dtype=int), np.zeros(50), p, 845, 'cpu')
        original_rng = torch.get_rng_state().clone()
        a, b = left.sample(0), right.sample(0)
        torch.testing.assert_close(original_rng, torch.get_rng_state(), atol=0, rtol=0)
        for key in ('positive_global', 'positive_local', 'negative_dual', 'negative_primal'):
            torch.testing.assert_close(getattr(a, key), getattr(b, key), atol=0, rtol=0)
        self.assertFalse(torch.equal(a.negative_dual, a.negative_primal))
        self.assertEqual(len(a.negative_dual.unique()), 7)
        self.assertTrue(torch.all(a.negative_dual >= 10))
        self.assertTrue(torch.all(a.negative_primal >= 10))
        self.assertTrue(torch.all(a.positive_global < 10))

    def test_proximal_center_and_outer_average_apply_to_weights_and_thresholds(self):
        labels = np.array([1, 1, 0, 0])
        p = dict(risk.parameters(), folds=1, inner_steps=2)
        state = risk.TailState(labels, np.zeros(4, dtype=int), np.zeros(4), p, 3, 'cpu')
        model = torch.nn.Linear(1, 1, bias=False).double()
        with torch.no_grad():
            model.weight.zero_()
        state.begin_stage(model)
        initial_s, initial_prime = state.s.clone(), state.prime.clone()
        with torch.no_grad():
            model.weight.fill_(1.)
            state.s.add_(1.)
            state.prime.add_(1.)
        self.assertFalse(state.accumulate(model))
        with torch.no_grad():
            model.weight.fill_(3.)
            state.s.add_(2.)
            state.prime.add_(2.)
        self.assertTrue(state.accumulate(model))
        self.assertEqual(float(model.weight.detach()), 2.)
        torch.testing.assert_close(state.s, initial_s+2., atol=0, rtol=0)
        torch.testing.assert_close(state.prime, initial_prime+2., atol=0, rtol=0)
        expected = 4/(2*p['theta_positive']*p['proximal_gamma'])
        self.assertAlmostEqual(float(state.parameter_penalty(model, 'full').detach()), expected)
        self.assertEqual(float(state.parameter_penalty(model, 'no_proximal').detach()), 0.)
        state.y.zero_()
        state.begin_stage(model)
        torch.testing.assert_close(state.y[state.positive], torch.ones(2, dtype=torch.float64), atol=0, rtol=0)
        self.assertEqual(float(state.parameter_penalty(model, 'full').detach()), 0.)
        self.assertEqual(state.completed_stages, 1)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
