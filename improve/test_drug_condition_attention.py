"""Checks that protect the interpretation of the attention ablation."""
import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData

from improve.drug_condition_attention import (
    AttentionModel, NeighbourBranch, masked_weights, rna_neighbour_masks)


class TinyBase(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(2 * width, 1)

    def encoder(self, x, edges):
        return x

    def decoder(self, z, pairs):
        return self.linear(torch.cat([z['miRNA'][pairs[0]], z['drug'][pairs[1]]], -1)).view(-1)


class AttentionSemanticsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.z = {'miRNA': torch.randn(4, 8), 'lncRNA': torch.randn(3, 8),
                  'drug': torch.randn(3, 8)}
        self.relations = [('miRNA', 'MiMi', 'miRNA'), ('lncRNA', 'LncMi', 'miRNA')]
        self.masks = [~torch.eye(4, dtype=torch.bool),
                      torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 0], [0, 1, 1]], dtype=torch.bool)]
        self.pairs = torch.tensor([[0, 0, 2, 3], [0, 1, 2, 0]])

    def branch(self, mode):
        return NeighbourBranch(8, 'miRNA', self.relations, self.masks, mode, width=4)

    def test_masked_and_empty_neighbours(self):
        logits = torch.tensor([[1., 2., 999.], [4., 5., 6.]], requires_grad=True)
        mask = torch.tensor([[True, True, False], [False, False, False]])
        weights = masked_weights(logits, mask)
        torch.testing.assert_close(weights[0, :2], torch.tensor([1., 2.]).softmax(0))
        self.assertEqual(weights[0, 2].item(), 0.)
        self.assertEqual(weights[1].abs().sum().item(), 0.)
        (weights * torch.arange(3.)).sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[0, 2].item(), 0.)
        self.assertEqual(logits.grad[1].abs().sum().item(), 0.)

    def test_only_rna_edges_define_neighbour_masks(self):
        data = HeteroData()
        for nt, x in self.z.items():
            data[nt].x = x
        data[self.relations[0]].edge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])
        data[self.relations[1]].edge_index = torch.tensor([[0, 1], [0, 1]])
        data['drug', 'rev_MiDrug', 'miRNA'].edge_index = torch.tensor([[0, 1], [0, 1]])
        rels, first = rna_neighbour_masks(data, 'miRNA')
        data['drug', 'rev_MiDrug', 'miRNA'].edge_index = torch.tensor([[2, 0, 1], [3, 2, 0]])
        _, second = rna_neighbour_masks(data, 'miRNA')
        self.assertEqual(rels, self.relations)
        self.assertFalse(first[0].diagonal().any())
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a, b))

    def test_only_conditioned_attention_changes_with_candidate_drug(self):
        for mode in ('uniform', 'node_attention', 'drug_attention'):
            branch = self.branch(mode)
            _, weights = branch.pool(branch.prepare(self.z), self.pairs, return_weights=True)
            difference = sum((w[0] - w[1]).abs().sum().item() for w in weights)
            if mode == 'drug_attention':
                self.assertGreater(difference, 1e-5)
            else:
                self.assertEqual(difference, 0.)
            self.assertTrue(all(torch.isfinite(w).all() for w in weights))
            self.assertEqual(weights[1][2].sum().item(), 0.)

    def test_zero_gate_recovers_base_and_can_learn(self):
        base = TinyBase(8)
        model = AttentionModel(base, self.relations, self.masks, 'drug_attention', 8, 'miRNA', 4)
        actual = model.decode(model.prepare(self.z, {}), self.pairs)
        expected = base.decoder(self.z, self.pairs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().sum().backward()
        self.assertGreater(model.branch.gate.grad.abs().item(), 0.)
        model.zero_grad()
        model.branch.gate.data.fill_(0.3)
        model.decode(model.prepare(self.z, {}), self.pairs).square().sum().backward()
        self.assertGreater(model.branch.query_d.weight.grad.abs().sum().item(), 0.)

    def test_chunking_preserves_loss_and_gradients(self):
        full = AttentionModel(TinyBase(8), self.relations, self.masks, 'drug_attention', 8, 'miRNA', 4)
        full.branch.gate.data.fill_(0.3)
        chunks = copy.deepcopy(full)
        y = torch.tensor([1., 0., 1., 0.])
        score = full.decode(full.prepare(self.z, {}), self.pairs)
        loss = F.binary_cross_entropy_with_logits(score, y)
        loss.backward()
        prepared = chunks.prepare(self.z, {})
        for i in range(2):
            selection = slice(2 * i, 2 * i + 2)
            score = chunks.decode(prepared, self.pairs[:, selection])
            part = F.binary_cross_entropy_with_logits(score, y[selection], reduction='sum') / len(y)
            part.backward(retain_graph=i == 0)
        for a, b in zip(full.parameters(), chunks.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
