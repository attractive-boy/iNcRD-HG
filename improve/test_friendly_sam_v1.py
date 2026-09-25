"""Closed-form optimizer checks and exact two-pass stochastic null."""
from copy import deepcopy
import unittest

import torch
from torch import nn

from improve import friendly_sam_v1 as sam


class Quadratic(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([.7, -.3], dtype=torch.float64))


class FriendlySAMTests(unittest.TestCase):
    def test_two_step_closed_form_and_clean_gradient_ema(self):
        hessian = torch.tensor([[3., .2], [.2, 1.]], dtype=torch.float64)
        target = torch.tensor([-.2, .4], dtype=torch.float64)
        for kind in sam.KINDS:
            model = Quadratic()
            optimizer = torch.optim.SGD(model.parameters(), lr=.04)
            stepper = sam.SharpnessStep(model, optimizer, kind, rho=.2, decay=.6, sigma=1., clip=1e6)
            expected = model.theta.detach().clone()
            previous = None
            def closure():
                error = model.theta - target
                return .5 * error @ hessian @ error
            for index in range(2):
                gradient = hessian @ (expected - target)
                direction = gradient - previous if kind == 'full' and previous is not None else gradient
                next_ema = gradient.clone() if previous is None else .6 * previous + .4 * gradient
                radius = 0. if kind == 'erm' else .2
                perturbation = radius * direction / direction.norm()
                expected -= .04 * (hessian @ (expected + perturbation - target))
                result = stepper.step(closure)
                torch.testing.assert_close(model.theta, expected, atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(stepper.momentum['theta'], next_ema, atol=1e-12, rtol=1e-12)
                self.assertAlmostEqual(result['perturbation_norm'], radius, places=12)
                previous = next_ema
            self.assertEqual(stepper.gradient_evaluations, 4)

    def test_zero_radius_matches_adamw_and_consumes_one_stochastic_draw(self):
        torch.manual_seed(884)
        model = nn.Linear(3, 1).double()
        baseline = deepcopy(model)
        x = torch.randn(9, 3, dtype=torch.float64)
        y = torch.randn(9, 1, dtype=torch.float64)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
        plain = torch.optim.AdamW(baseline.parameters(), lr=.003, weight_decay=.001)
        stepper = sam.SharpnessStep(model, optimizer, 'full', rho=0.)
        for _ in range(3):
            masks = []
            def closure():
                mask = torch.bernoulli(torch.full_like(x, .75)) / .75
                masks.append(mask.clone())
                return (model(x * mask) - y).square().mean()
            initial_rng = torch.random.get_rng_state()
            stepper.step(closure)
            final_rng = torch.random.get_rng_state()
            self.assertEqual(len(masks), 2)
            torch.testing.assert_close(masks[0], masks[1], atol=0, rtol=0)
            torch.random.set_rng_state(initial_rng)
            plain.zero_grad(set_to_none=True)
            mask = torch.bernoulli(torch.full_like(x, .75)) / .75
            (baseline(x * mask) - y).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(baseline.parameters(), 5.)
            plain.step()
            self.assertTrue(torch.equal(torch.random.get_rng_state(), final_rng))
            for a, b in zip(model.parameters(), baseline.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_failed_second_pass_restores_parameters_and_momentum(self):
        model = Quadratic()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
        stepper = sam.SharpnessStep(model, optimizer, 'full')
        original = model.theta.detach().clone()
        calls = 0
        def closure():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('injected second-pass failure')
            return model.theta.square().sum()
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            stepper.step(closure)
        torch.testing.assert_close(model.theta, original, atol=0, rtol=0)
        self.assertEqual(stepper.momentum, {})
        self.assertEqual(dict(optimizer.state), {})


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
