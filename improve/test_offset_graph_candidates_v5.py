import unittest
from unittest.mock import patch

import torch
from torch import nn

from improve.offset_graph_candidates_v5 import OffsetCandidate, fit


class ToyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.rna_degree = nn.Linear(2, 1)
        self.drug_degree = nn.Linear(2, 1)
        self.interaction = nn.Linear(2, 1)

    def forward(self, xs, graph, pairs):
        return self.interaction(xs[pairs[0]]).squeeze(-1) + self.rna_degree(xs[pairs[0]]).squeeze(-1)


class OffsetCandidateTests(unittest.TestCase):
    def test_offsets_are_fixed_additive_and_query_order_equivariant(self):
        core = ToyCore()
        model = OffsetCandidate(core)
        x = torch.randn(4, 2)
        offsets = torch.tensor([.2, -.3, .1, .9], dtype=torch.float64)
        intercept = torch.tensor(.7, dtype=torch.float64)
        pairs = torch.tensor([[0, 1, 2], [3, 2, 1]])
        actual = model(x, (None, offsets, intercept), pairs)
        expected = core(x, None, pairs).double() + offsets[pairs[0]] + offsets[pairs[1]] + intercept
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual, model(x, (None, offsets, intercept), pairs.flip(1)).flip(0))
        actual.sum().backward()
        for head in (core.rna_degree, core.drug_degree):
            for p in head.parameters():
                self.assertFalse(p.requires_grad)
                self.assertIsNone(p.grad)
                self.assertEqual(torch.count_nonzero(p).item(), 0)
        self.assertGreater(core.interaction.weight.grad.abs().sum().item(), 0)

    def test_fixed_fit_rejects_query_before_model_construction(self):
        for field in ('pairs_val', 'y_val', 'offset_val', 'anchor_666'):
            with self.subTest(field=field), patch('improve.offset_graph_candidates_v5.setup') as setup:
                with self.assertRaisesRegex(ValueError, 'training inputs only'):
                    fit({field: object()}, {}, 'full', .001, 0, 'cpu', fixed_epochs=2)
                setup.assert_not_called()


if __name__ == '__main__':
    unittest.main()
