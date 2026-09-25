import unittest
from unittest.mock import patch

import torch
from torch import nn

from improve.offset_graph_candidates_v10 import OffsetCandidate, fit


class ToyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.rna_degree = nn.Linear(2, 1)
        self.drug_degree = nn.Linear(2, 1)
        self.interaction = nn.Linear(2, 1)

    def forward(self, xs, graph, pairs):
        return self.interaction(xs[pairs[0]]).squeeze(-1) + self.rna_degree(xs[pairs[0]]).squeeze(-1)


class JointOffsetTests(unittest.TestCase):
    def test_task_specific_offsets_fixed_and_additive(self):
        core = ToyCore()
        model = OffsetCandidate(core)
        x = torch.randn(4, 2)
        offsets = torch.tensor([.2, -.3, .1, .9], dtype=torch.float64)
        auxiliary = torch.tensor([.7, .8, -.1, -.4], dtype=torch.float64)
        packed = (None, offsets, torch.tensor(.7), auxiliary, torch.tensor(-.3))
        pairs = torch.tensor([[0, 1, 2], [3, 2, 1]])
        raw = core(x, None, pairs)
        actual = model(x, packed, pairs)
        torch.testing.assert_close(actual, raw.double() + offsets[pairs[0]] + offsets[pairs[1]] + packed[2])
        torch.testing.assert_close(model.compose(raw, packed, pairs, secondary=True),
                                   raw.double() + auxiliary[pairs[0]] + auxiliary[pairs[1]] + packed[4])
        torch.testing.assert_close(actual, model(x, packed, pairs.flip(1)).flip(0))
        actual.sum().backward()
        for head in (core.rna_degree, core.drug_degree):
            for p in head.parameters():
                self.assertFalse(p.requires_grad)
                self.assertIsNone(p.grad)
                self.assertEqual(torch.count_nonzero(p).item(), 0)
        self.assertGreater(core.interaction.weight.grad.abs().sum().item(), 0)

    def test_fixed_fit_rejects_both_tasks_queries_before_model_construction(self):
        for field in ('pairs_val', 'y_val', 'offset_val', 'anchor_666', 'secondary_y_val', 'secondary_pairs_val'):
            with self.subTest(field=field), patch('improve.offset_graph_candidates_v10.setup') as setup:
                with self.assertRaisesRegex(ValueError, 'training inputs only'):
                    fit({field: object()}, {}, 'full', .001, 0, 'cpu', fixed_epochs=2)
                setup.assert_not_called()


if __name__ == '__main__':
    unittest.main()
