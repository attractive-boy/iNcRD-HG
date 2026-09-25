"""Independent singular-value oracle, update controls and optimizer state."""
from copy import deepcopy
import unittest

import torch
from torch import nn

from improve import spectral_muon_v1 as spectral


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.embedding = nn.Embedding(5, 3)
        self.norm = nn.LayerNorm(3)
        self.head = nn.Linear(3, 1)


class SpectralOptimizerTests(unittest.TestCase):
    def test_five_step_iteration_matches_singular_value_oracle(self):
        torch.manual_seed(247)
        for shape in ((3, 7), (7, 3), (4, 4)):
            g = torch.randn(*shape, dtype=torch.float64)
            u, s, vh = torch.linalg.svd(g, full_matrices=False)
            values = s / (s.norm() + 1e-7)
            for _ in range(5):
                values = 3.4445 * values - 4.7750 * values ** 3 + 2.0315 * values ** 5
            expected = (u * values) @ vh
            actual = spectral.newton_schulz(g)
            torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
            torch.testing.assert_close(actual.T, spectral.newton_schulz(g.T), atol=3e-12, rtol=3e-12)
        zero = torch.zeros(4, 7, dtype=torch.float64)
        torch.testing.assert_close(spectral.newton_schulz(zero), zero, atol=0, rtol=0)

    def test_control_keeps_norm_while_removing_singular_reweighting(self):
        g = torch.diag(torch.tensor([10., 1., .1], dtype=torch.float64))
        full = spectral.matrix_direction(g, 'full')
        control = spectral.matrix_direction(g, 'frobenius')
        torch.testing.assert_close(full.norm(), control.norm(), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(control / control.norm(), g / g.norm(), atol=1e-12, rtol=1e-12)
        self.assertGreater((full / full.norm() - control / control.norm()).abs().max().item(), .05)
        torch.testing.assert_close(spectral.matrix_direction(g * 0, 'frobenius'), g * 0, atol=0, rtol=0)

    def test_parameter_dispatch_and_backup_match_adamw(self):
        torch.manual_seed(29)
        model = ToyModel().double()
        other = deepcopy(model)
        opt = spectral.SpectralMuon(model, lr=.003)
        baseline = torch.optim.AdamW(other.parameters(), lr=.003, weight_decay=.001)
        self.assertEqual(spectral.matrix_parameter_names(model), ('linear.weight',))
        for _ in range(4):
            for p, q in zip(model.parameters(), other.parameters()):
                g = torch.randn_like(p); p.grad = g.clone(); q.grad = g.clone()
            opt.step(); baseline.step()
            for (name, p), q in zip(model.named_parameters(), other.parameters()):
                if name != 'linear.weight':
                    torch.testing.assert_close(p, q, atol=1e-14, rtol=1e-14)

    def test_optimizer_resume_and_nonfinite_failure_preserve_state(self):
        torch.manual_seed(82)
        model = ToyModel().double()
        opt = spectral.SpectralMuon(model, lr=.001)
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        opt.step()
        second = deepcopy(model)
        replay = spectral.SpectralMuon(second, lr=.001)
        replay.load_state_dict(deepcopy(opt.state_dict()))
        for p, q in zip(model.parameters(), second.parameters()):
            g = torch.randn_like(p); p.grad = g.clone(); q.grad = g.clone()
        opt.step(); replay.step()
        for p, q in zip(model.parameters(), second.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)
        saved = deepcopy(model.state_dict())
        model.head.bias.grad.fill_(float('nan'))
        with self.assertRaisesRegex(FloatingPointError, 'finite dense'):
            opt.step()
        for name, p in model.state_dict().items():
            torch.testing.assert_close(p, saved[name], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
