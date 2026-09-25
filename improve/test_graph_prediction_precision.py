"""Regression coverage for inference determinism and posterior tail scores."""
import unittest

import numpy as np
import torch

from improve.graph_prediction_precision import deterministic_evaluation, geometry_prediction


class FixedLogits:
    def eval(self):
        return self

    def __call__(self, xs, graph, pairs):
        return xs[pairs[0]] + graph


class PrecisionTest(unittest.TestCase):
    def test_inverse_logit_at_probability_tails(self):
        logits = torch.tensor([-15.5, -14., 0., 14., 15.5], dtype=torch.float32)
        pairs = torch.stack((torch.arange(len(logits)), torch.zeros(len(logits), dtype=torch.long)))
        result = geometry_prediction(FixedLogits(), logits, [0.], pairs)
        np.testing.assert_allclose(result.numpy(), logits.numpy(), atol=1e-8, rtol=1e-8)
        # The old float32 probability route loses more than 0.1 logit here.
        self.assertGreater(float((logits.sigmoid().logit()-logits).abs().max()), .1)

    def test_restore_global_flags_after_error(self):
        initial = (torch.are_deterministic_algorithms_enabled(),
                   torch.is_deterministic_algorithms_warn_only_enabled(),
                   torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            with deterministic_evaluation():
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                raise RuntimeError('sentinel')
        after = (torch.are_deterministic_algorithms_enabled(),
                 torch.is_deterministic_algorithms_warn_only_enabled(),
                 torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        self.assertEqual(initial, after)


if __name__ == '__main__':
    unittest.main()
