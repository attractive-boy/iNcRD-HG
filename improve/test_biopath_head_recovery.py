"""Check the specific training failure that made the old head constant."""
import unittest

import torch

from improve.biopath_head_recovery import LivePathResidual


class HeadRecoveryTest(unittest.TestCase):
    def test_zero_start(self):
        model = LivePathResidual(4, width=3, depth=1)
        x = torch.randn(7, 6)
        torch.testing.assert_close(model.mlp(x), torch.zeros(7, 1))

    def test_gradient_survives_all_negative_hidden_preactivations(self):
        model = LivePathResidual(4, width=3, depth=1)
        with torch.no_grad():
            model.mlp[0].weight.fill_(0.1)
            model.mlp[0].bias.fill_(-2.)
            model.mlp[2].weight.fill_(0.2)
        x = torch.ones(7, 6, requires_grad=True)
        self.assertTrue(bool((model.mlp[0](x) < 0).all()))
        model.mlp(x).sum().backward()
        self.assertTrue(bool((x.grad.abs() > 0).all()))
        self.assertGreater(float(model.mlp[0].weight.grad.norm()), 0.)


if __name__ == '__main__':
    unittest.main()
