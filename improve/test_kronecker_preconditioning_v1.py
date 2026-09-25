"""Numerical contracts for the new matrix optimizer; CPU only."""
import copy
import unittest

import torch
from torch import nn

from improve import kronecker_preconditioning_v1 as core


class KroneckerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.rng = torch.Generator(device='cpu').manual_seed(39)

    def rand(self, *shape):
        return torch.randn(*shape, generator=self.rng, dtype=torch.float64)

    def test_coupled_targets_match_explicit_inverse_covariance(self):
        g = self.rand(3, 4)
        ql, qr = torch.linalg.qr(self.rand(3, 3)).Q, torch.linalg.qr(self.rand(4, 4)).Q
        il, ir = torch.tensor([.4, 2., 1.], dtype=g.dtype), torch.tensor([1., .7, 3., 2.], dtype=g.dtype)
        left, right, dl, dr = core.covariance_targets(g, ql, qr, il, ir, 'full')
        pl, pr = ql @ torch.diag(il.square()) @ ql.T, qr @ torch.diag(ir.square()) @ qr.T
        torch.testing.assert_close(left, g @ pr @ g.T / 4, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(right, g.T @ pl @ g / 3, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(dl, torch.diag(ql.T @ left @ ql), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(dr, torch.diag(qr.T @ right @ qr), atol=1e-12, rtol=1e-12)
        ml = ql @ torch.diag(il) @ ql.T
        mr = qr @ torch.diag(ir) @ qr.T
        # Row-major vec uses left kron right, rather than the column-major form.
        expected = (torch.kron(ml.contiguous(), mr.contiguous()) @ g.flatten()).view_as(g)
        actual = ql @ ((ql.T @ g @ qr) * il[:, None] * ir) @ qr.T
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_scalar_control_keeps_scale_but_removes_direction_coupling(self):
        g = self.rand(3, 4)
        ql, qr = torch.linalg.qr(self.rand(3, 3)).Q, torch.linalg.qr(self.rand(4, 4)).Q
        il, ir = g.new_full((3,), 2.), g.new_full((4,), 3.)
        a = core.covariance_targets(g, ql, qr, il, ir, 'full')
        b = core.covariance_targets(g, ql, qr, il, ir, 'scalar_coupling')
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-12)
        il[0], ir[0] = .2, .1
        a = core.covariance_targets(g, ql, qr, il, ir, 'full')
        b = core.covariance_targets(g, ql, qr, il, ir, 'scalar_coupling')
        self.assertGreater(float((a[0]-b[0]).norm()), .01)
        torch.testing.assert_close(b[0], (g @ g.T) * ir.square().mean() / 4)

    def test_adamw_matches_torch_after_declared_initialization_gradient(self):
        model = nn.Linear(4, 3).double()
        reference = copy.deepcopy(model)
        ours = core.optimizer(model, 'adamw', .003, .001, core.parameters())
        theirs = torch.optim.AdamW(reference.parameters(), lr=.003, weight_decay=.001,
                                  betas=(.9, .98), eps=1e-8, foreach=False)
        original = copy.deepcopy(model.state_dict())
        for step in range(26):
            for x, y in zip(model.parameters(), reference.parameters()):
                x.grad = self.rand(*x.shape)
                y.grad = x.grad.clone()
            ours.step()
            if step:
                theirs.step()
            else:
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, original[key], rtol=0, atol=0)
            for x, y in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(x, y, rtol=1e-13, atol=1e-13)
        self.assertEqual(ours.diagnostics()['parameter_updates_min'], 25)

    def test_kl_update_uses_old_factors_and_preserves_momentum_at_qr(self):
        model = nn.Linear(4, 3, bias=False).double()
        optimizer = core.optimizer(model, 'full', .002, 0., core.parameters())
        model.weight.grad = self.rand(3, 4)
        optimizer.step()
        state = optimizer.state[model.weight]
        state['step'] = 9
        state['left_inverse'] = self.rand(3).abs() + .5
        state['right_inverse'] = self.rand(4).abs() + .5
        state['exp_avg'] = self.rand(3, 4)
        before = copy.deepcopy(state)
        weight = model.weight.detach().clone()
        gradient = self.rand(3, 4)
        model.weight.grad = gradient
        targets = core.covariance_targets(gradient, before['left_basis'], before['right_basis'],
            before['left_inverse'], before['right_inverse'], 'full')
        expected_m = .9 * before['exp_avg'] + .1 * gradient
        scale = before['left_inverse'][:, None] * before['right_inverse']
        scale /= 1 + 1e-8 * scale
        ql, qr = before['left_basis'], before['right_basis']
        expected_weight = weight - .002 * ql @ ((ql.T @ expected_m @ qr) * scale) @ qr.T
        optimizer.step()
        torch.testing.assert_close(model.weight, expected_weight, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(state['exp_avg'], expected_m, atol=1e-12, rtol=1e-12)
        for side, covariance, diagonal in zip(('left', 'right'), targets[:2], targets[2:]):
            torch.testing.assert_close(state[side+'_covariance'],
                .98 * before[side+'_covariance'] + .02 * covariance, atol=1e-12, rtol=1e-12)
            expected_inverse = (.98 / before[side+'_inverse'].square() + .02*diagonal).rsqrt().clamp(max=10)
            torch.testing.assert_close(state[side+'_inverse'], expected_inverse, atol=1e-12, rtol=1e-12)
            q = state[side+'_basis']
            torch.testing.assert_close(q.T @ q, torch.eye(len(q), dtype=q.dtype), atol=1e-12, rtol=1e-12)
        self.assertEqual(state['basis_updates'], 1)

    def test_all_arms_resume_across_basis_updates_and_keep_vector_trajectory(self):
        original = nn.Linear(4, 3).double()
        models = {k: copy.deepcopy(original) for k in core.KINDS}
        optimizers = {k: core.optimizer(v, k, .001, .001, core.parameters()) for k, v in models.items()}
        resumed_model, resumed_opt = {}, {}
        for step in range(24):
            gradients = [self.rand(*x.shape) for x in original.parameters()]
            for kind in core.KINDS:
                for p, g in zip(models[kind].parameters(), gradients):
                    p.grad = g.clone()
                optimizers[kind].step()
                if step == 12:
                    m = copy.deepcopy(models[kind])
                    o = core.optimizer(m, kind, .001, .001, core.parameters())
                    o.load_state_dict(copy.deepcopy(optimizers[kind].state_dict()))
                    resumed_model[kind], resumed_opt[kind] = m, o
                elif step > 12:
                    for p, g in zip(resumed_model[kind].parameters(), gradients):
                        p.grad = g.clone()
                    resumed_opt[kind].step()
                    for x, y in zip(models[kind].parameters(), resumed_model[kind].parameters()):
                        torch.testing.assert_close(x, y, atol=0, rtol=0)
                torch.testing.assert_close(models[kind].bias, models['full'].bias, rtol=0, atol=0)
        for kind, o in optimizers.items():
            d = o.diagnostics()
            self.assertTrue(d['state_finite'])
            self.assertEqual(d['parameter_updates_min'], 23)
            self.assertEqual(d['basis_updates_min'], 0 if kind == 'adamw' else 2)

    def test_invalid_gradient_changes_no_parameter_or_state(self):
        model = nn.Linear(4, 3).double()
        optimizer = core.optimizer(model, 'full', .001, .001, core.parameters())
        old = copy.deepcopy(model.state_dict())
        model.weight.grad = self.rand(3, 4)
        model.bias.grad = torch.full_like(model.bias, torch.nan)
        with self.assertRaises(FloatingPointError):
            optimizer.step()
        self.assertFalse(optimizer.state)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, old[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
