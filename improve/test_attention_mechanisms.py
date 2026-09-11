"""Interpretation-critical checks for the controlled attention experiments."""
import copy
import unittest

import torch
from torch.nn import functional as F
from torch_geometric.data import HeteroData

from improve.attention_mechanisms import (
    ARMS, ChannelBranch, LayerBranch, MechanismModel, NeighbourBranch,
    capture_first_layer, clean_splits, norm, topk_mask,
)
from improve.drug_condition_attention import make_base, masked_weights, set_seed


def tiny_data():
    set_seed(41)
    data = HeteroData()
    data['miRNA'].x = torch.randn(4, 6)
    data['drug'].x = torch.randn(3, 5)
    data['miRNA', 'MiMi', 'miRNA'].edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]])
    data['drug', 'DrugDrug', 'drug'].edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    key = ('miRNA', 'MiDrug', 'drug')
    data[key].edge_index = torch.tensor([[0, 1], [0, 1]])
    data['drug', 'rev_MiDrug', 'miRNA'].edge_index = data[key].edge_index.flip(0)
    data[key].edge_label_index = torch.tensor([[0, 1, 2, 3], [0, 1, 0, 1]])
    data[key].edge_label = torch.tensor([1., 1., 0., 0.])
    return data, key


class MechanismChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_overlap_removed_from_training_labels_and_both_graph_directions(self):
        data, key = tiny_data()
        splits = {f: copy.deepcopy(data) for f in ('train', 'val', 'test')}
        splits['val'][key].edge_label_index = torch.tensor([[0, 2], [0, 1]])
        splits['val'][key].edge_label = torch.tensor([1., 0.])
        splits['test'][key].edge_label_index = torch.tensor([[3, 0], [2, 2]])
        splits['test'][key].edge_label = torch.tensor([1., 0.])
        clean, removed = clean_splits(splits, key)
        self.assertEqual(removed['train'], [dict(pair=[0, 0], label=1)])
        self.assertEqual(clean['train'][key].edge_index.tolist(), [[1], [1]])
        self.assertEqual(clean['train']['drug', 'rev_MiDrug', 'miRNA'].edge_index.tolist(), [[1], [1]])
        self.assertTrue(torch.equal(clean['val'][key].edge_label_index, splits['val'][key].edge_label_index))
        self.assertEqual(data[key].edge_index.size(1), 2)
        splits['val'][key].edge_label = torch.tensor([0., 1.])
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            clean_splits(splits, key)

    def test_topk_excludes_missing_edges_handles_empty_rows_and_has_finite_gradients(self):
        logits = torch.tensor([[1., 5., 999., 3.], [3., 2., 1., 0.]], requires_grad=True)
        support = torch.tensor([[True, True, False, True], [False, False, False, False]])
        selected = topk_mask(logits, support, 2)
        self.assertEqual(selected.tolist(), [[False, True, False, True], [False] * 4])
        weights = masked_weights(logits, selected)
        self.assertAlmostEqual(float(weights[0].sum()), 1.)
        self.assertEqual(float(weights[1].sum()), 0.)
        (weights * torch.arange(4)).sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad[~selected].abs().sum()), 0.)
        self.assertTrue(torch.equal(topk_mask(logits, support, 9), support))

    def test_fx_capture_matches_real_first_layer_and_preserves_base_output(self):
        data, key = tiny_data()
        config = dict(hidden=8, aggr='mean', rna='miRNA')
        base = make_base(data, config, 42, 'cpu')
        expected = base.encoder(data.x_dict, data.edge_index_dict)
        messages = []
        for relation, edges in data.edge_index_dict.items():
            if relation[2] == 'miRNA':
                conv = base.encoder.get_submodule('convs.0.' + '__'.join(relation))
                messages.append(conv((data[relation[0]].x, data[relation[2]].x), edges))
        first_expected = F.leaky_relu(torch.stack(messages).mean(0), 0.1)
        capture_first_layer(base.encoder, data.node_types)
        final, first = base.encoder(data.x_dict, data.edge_index_dict)
        for nt in data.node_types:
            torch.testing.assert_close(final[nt], expected[nt], rtol=0, atol=0)
        torch.testing.assert_close(first['miRNA'], first_expected)

    def test_every_arm_starts_at_identical_predictions_and_its_branch_learns(self):
        data, key = tiny_data()
        config = dict(hidden=8, aggr='mean', rna='miRNA')
        for arm in ARMS:
            with self.subTest(arm=arm):
                base = make_base(data, config, 42, 'cpu')
                expected = base(data.x_dict, data.edge_index_dict, data[key].edge_label_index).detach()
                set_seed(100)
                model = MechanismModel(base, data, config, arm, width=4, k=2, seed=42)
                actual = model(data.x_dict, data.edge_index_dict, data[key].edge_label_index)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                if model.branch is None:
                    continue
                before = {k: v.clone() for k, v in model.branch.named_parameters()}
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                for _ in range(4):
                    optimizer.zero_grad()
                    loss = F.binary_cross_entropy_with_logits(
                        model(data.x_dict, data.edge_index_dict, data[key].edge_label_index), data[key].edge_label)
                    loss.backward()
                    for p in model.parameters():
                        if p.grad is not None:
                            self.assertTrue(torch.isfinite(p.grad).all())
                    optimizer.step()
                changed = [k for k, v in model.branch.named_parameters() if not torch.equal(before[k], v)]
                self.assertTrue(changed)
                if arm == 'layer_attention':
                    self.assertTrue(any(k.startswith('scorers.') for k in changed))
                if arm in ('neighbour_topk', 'neighbour_softmax', 'neighbour_randomk'):
                    self.assertTrue(any(k.startswith('query.') for k in changed))

    def test_channel_control_matches_parameter_count_and_attention_is_node_dependent(self):
        a = ChannelBranch(8, ['miRNA'], 'channel_attention')
        b = ChannelBranch(8, ['miRNA'], 'channel_mlp')
        self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))
        torch.nn.init.normal_(a.gates['miRNA'][-1].weight)
        z = {'miRNA': torch.randn(4, 8)}
        _, weights = a({}, None, z)
        self.assertGreater(float(weights['channel_miRNA'].std(0).mean()), 0.)
        self.assertTrue(((weights['channel_miRNA'] >= 0) & (weights['channel_miRNA'] <= 2)).all())

    def test_random_k_is_reproducible_and_has_the_same_per_node_budget(self):
        relation = [('miRNA', 'MiMi', 'miRNA')]
        support = ~torch.eye(10, dtype=torch.bool)
        support[-1] = False
        branches = [NeighbourBranch(8, 'miRNA', relation, [support], 'neighbour_randomk', 4, 3, s)
                    for s in (17, 17, 18)]
        self.assertTrue(torch.equal(branches[0].random_0, branches[1].random_0))
        self.assertFalse(torch.equal(branches[0].random_0, branches[2].random_0))
        self.assertTrue(torch.equal(branches[0].random_0.sum(-1), support.sum(-1).clamp_max(3)))
        self.assertFalse((branches[0].random_0 & ~support).any())


if __name__ == '__main__':
    unittest.main()
