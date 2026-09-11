"""Checks preserving the interpretation of propagation-attention controls."""
import copy
import unittest

import torch
from torch.nn import functional as F
from torch_geometric.nn import GeneralConv

from improve.drug_condition_attention import make_base, set_seed
from improve.propagation_attention import (
    ARMS, PRIMARY, EdgeGateConv, GATv2WithRoot, PropagationModel,
    RelationFusion, gate_weights,
)
from improve.test_attention_mechanisms import tiny_data


class PropagationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_normalisation_preserves_degree_or_unit_mass_with_duplicate_edges(self):
        scores = torch.tensor([0.2, -1., 2., 4., -0.4], requires_grad=True)
        destination = torch.tensor([0, 0, 0, 2, 2])
        for preserve, expected in ((True, [3., 0., 2., 0.]), (False, [1., 0., 1., 0.])):
            weights = gate_weights(scores, destination, 4, preserve)
            total = weights.new_zeros(4).index_add(0, destination, weights)
            torch.testing.assert_close(total, torch.tensor(expected))
            self.assertTrue((weights > 0).all())
            (weights * torch.arange(5)).sum().backward()
            self.assertTrue(torch.isfinite(scores.grad).all())
            self.assertGreater(float(scores.grad.abs().sum()), 0.)
            scores.grad = None
        unit = gate_weights(torch.zeros(5), destination, 4, True)
        torch.testing.assert_close(unit, torch.ones(5), rtol=0, atol=0)
        empty = gate_weights(torch.empty(0), torch.empty(0, dtype=torch.long), 4, True)
        self.assertEqual(empty.numel(), 0)

    def test_degree_gate_recovers_original_messages_root_and_gradients(self):
        set_seed(13)
        original = GeneralConv((5, 6), 8)
        wrapped = EdgeGateConv(original, 'edge_gate_degree', 4)
        source = torch.randn(3, 5, requires_grad=True)
        target = torch.randn(4, 6, requires_grad=True)
        edges = torch.tensor([[0, 0, 1, 2], [0, 0, 0, 2]])
        expected = original((source, target), edges)
        actual = wrapped((source, target), edges)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().sum().backward()
        a, b = source.grad.clone(), target.grad.clone()
        source.grad, target.grad = None, None
        expected.square().sum().backward()
        torch.testing.assert_close(source.grad, a)
        torch.testing.assert_close(target.grad, b)
        torch.testing.assert_close(actual[1], original.lin_self(target)[1])

    def test_uniform_gatv2_retains_root_and_matches_mean_messages(self):
        set_seed(15)
        original = GeneralConv((5, 6), 8)
        gate = EdgeGateConv(original, 'edge_uniform_mean', 4)
        uniform = GATv2WithRoot(original, 'gatv2_uniform', heads=4)
        learned = GATv2WithRoot(original, 'gatv2_attention', heads=4)
        source, target = torch.randn(3, 5), torch.randn(4, 6)
        edges = torch.tensor([[0, 0, 1, 2], [0, 0, 0, 2]])
        expected = gate((source, target), edges)
        torch.testing.assert_close(uniform((source, target), edges), expected)
        torch.testing.assert_close(learned((source, target), edges), expected)
        torch.testing.assert_close(expected[1], original.lin_self(target)[1])
        self.assertEqual(uniform.gat.last_weights.shape, (4, 4))

    def test_relation_weights_are_global_or_node_dependent_as_declared(self):
        set_seed(19)
        messages = [torch.randn(5, 8) for _ in range(3)]
        global_attention = RelationFusion(8, 'relation_global', 4)
        node_attention = RelationFusion(8, 'relation_node', 4)
        with torch.no_grad():
            global_attention.scorer[-1].weight.fill_(0.7)
            node_attention.load_state_dict(global_attention.state_dict())
        global_attention(messages)
        node_attention(messages)
        self.assertEqual(float(global_attention.last_weights.std(0).sum()), 0.)
        self.assertGreater(float(node_attention.last_weights.std(0).sum()), 0.)
        torch.testing.assert_close(node_attention.last_weights.sum(-1), torch.ones(5))
        expected = node_attention(messages)
        permutation = torch.tensor([3, 0, 4, 1, 2])
        actual = node_attention([x[permutation] for x in messages])
        torch.testing.assert_close(actual, expected[permutation])

    def test_fx_fusion_replacement_matches_independent_mean_encoder(self):
        data, key = tiny_data()
        for aggregation in ('min', 'mean'):
            config = dict(hidden=8, aggr=aggregation, rna='miRNA')
            original = make_base(data, config, 42, 'cpu')
            mean_config = dict(config, aggr='mean')
            reference = make_base(data, mean_config, 42, 'cpu')
            reference.load_state_dict(original.state_dict())
            expected = reference(data.x_dict, data.edge_index_dict, data[key].edge_label_index)
            for arm in ('relation_uniform', 'relation_global', 'relation_node'):
                model = PropagationModel(copy.deepcopy(original), data, config, arm, width=4)
                torch.testing.assert_close(model(data.x_dict, data.edge_index_dict,
                                                data[key].edge_label_index), expected)

    def test_all_arms_train_and_attention_parameters_receive_updates(self):
        data, key = tiny_data()
        config = dict(hidden=8, aggr='mean', rna='miRNA')
        for arm in ARMS:
            with self.subTest(arm=arm):
                base = make_base(data, config, 42, 'cpu')
                set_seed(51)
                model = PropagationModel(base, data, config, arm, width=4)
                before = {n: p.detach().clone() for n, p in model.named_parameters()}
                optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
                for _ in range(4):
                    optimizer.zero_grad()
                    score = model(data.x_dict, data.edge_index_dict, data[key].edge_label_index)
                    loss = F.binary_cross_entropy_with_logits(score, data[key].edge_label)
                    loss.backward()
                    for p in model.parameters():
                        if p.grad is not None:
                            self.assertTrue(torch.isfinite(p.grad).all())
                    optimizer.step()
                changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
                self.assertTrue(changed)
                if arm in ('relation_node', 'relation_global'):
                    self.assertTrue(any('scorer.2.weight' in n for n in changed))
                elif arm.startswith('edge_gate_'):
                    self.assertTrue(any('score.weight' in n for n in changed))
                elif arm == 'gatv2_attention':
                    self.assertTrue(any(n.endswith('gat.att') for n in changed))
                if arm in PRIMARY:
                    expected = model(data.x_dict, data.edge_index_dict, data[key].edge_label_index)
                    model.uniform(True)
                    altered = model(data.x_dict, data.edge_index_dict, data[key].edge_label_index)
                    self.assertTrue(torch.isfinite(altered).all())
                    model.uniform(False)
                    replay = model(data.x_dict, data.edge_index_dict, data[key].edge_label_index)
                    torch.testing.assert_close(replay, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
