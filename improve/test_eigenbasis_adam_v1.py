"""Numerical variance-factorization and unchanged-parent contracts."""
import copy
import math
import unittest

import torch
from torch import nn

from improve import eigenbasis_adam_v1 as core


class EigenbasisTests(unittest.TestCase):
    def setUp(self):
        self.rng = torch.Generator(device='cpu').manual_seed(822)

    def rand(self, *shape):
        return torch.randn(*shape, dtype=torch.float64, generator=self.rng)

    def test_factorization_preserves_marginals_and_rank_one_inputs(self):
        variance = self.rand(3, 4).square()
        result = core.separable_second_moment(variance)
        torch.testing.assert_close(result.sum(0), variance.sum(0), rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(result.sum(1), variance.sum(1), rtol=1e-14, atol=1e-14)
        self.assertEqual(int(torch.linalg.matrix_rank(result)), 1)
        rank_one = self.rand(3, 1).square() @ self.rand(1, 4).square()
        torch.testing.assert_close(core.separable_second_moment(rank_one), rank_one, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(core.separable_second_moment(torch.zeros_like(variance)),
                                   torch.zeros_like(variance), rtol=0, atol=0)

    def test_exact_parent_trajectories(self):
        for kind, parent_kind in (('full', 'soap'), ('adamw', 'adamw')):
            model = nn.Linear(4, 3).double()
            other = copy.deepcopy(model)
            ours = core.optimizer(model, kind, .003, .001, core.parameters())
            old = core.parent.optimizer(other, parent_kind, .003, .001, core.parameters())
            for _ in range(24):
                for x, y in zip(model.parameters(), other.parameters()):
                    x.grad = self.rand(*x.shape); y.grad = x.grad.clone()
                ours.step(); old.step()
                for x, y in zip(model.parameters(), other.parameters()):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_separable_update_matches_formula_and_resumes_across_qr(self):
        model = nn.Linear(4, 3, bias=False).double()
        optimizer = core.optimizer(model, 'separable_variance', .003, .001, core.parameters())
        model.weight.grad = self.rand(3, 4)
        optimizer.step()
        state = optimizer.state[model.weight]
        for _ in range(12):
            gradient = self.rand(3, 4)
            before = copy.deepcopy(state)
            weight = model.weight.detach().clone()
            step = before['step'] + 1
            ql, qr = before['left_basis'], before['right_basis']
            g = ql.T @ gradient @ qr
            v = .98*before['exp_avg_sq'] + .02*g.square()
            m = .9*before['exp_avg'] + .1*gradient
            vhat = v.sum(1, keepdim=True) * v.sum(0, keepdim=True) / v.sum()
            correction = math.sqrt(1-.98**step)/(1-.9**step)
            delta = (ql @ ((ql.T @ m @ qr)/(vhat.sqrt()+1e-8)) @ qr.T) * correction
            model.weight.grad = gradient
            optimizer.step()
            torch.testing.assert_close(model.weight, weight*(1-.003*.001)-.003*delta, rtol=1e-12, atol=1e-12)
        resumed = copy.deepcopy(model)
        other = core.optimizer(resumed, 'separable_variance', .003, .001, core.parameters())
        other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        for _ in range(12):
            model.weight.grad = self.rand(3, 4)
            resumed.weight.grad = model.weight.grad.clone()
            optimizer.step(); other.step()
            torch.testing.assert_close(model.weight, resumed.weight, rtol=0, atol=0)
        self.assertEqual(optimizer.diagnostics()['basis_updates_min'], 2)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
