"""Check that context bags preserve graph independence and matched controls."""
import unittest
import numpy as np
import torch
from torch_geometric.data import HeteroData

from improve.cell_context_mil import (
    context_features, make_groups, pack_context_graph, pool_context_logits)
from improve.network_feature_ablation import ExperimentModel


class CellContextMilTests(unittest.TestCase):
    def test_pooling_identity_and_order_invariance(self):
        x = torch.tensor([[-1., 0., 2.]]).repeat(5, 1)
        for mode in ('mean', 'softor'):
            torch.testing.assert_close(pool_context_logits(x, mode), x[0])
        varied = torch.randn(5, 9)
        permuted = torch.cat([varied[:1], varied[1:][torch.tensor([2, 0, 3, 1])]])
        torch.testing.assert_close(pool_context_logits(varied, 'softor'), pool_context_logits(permuted, 'softor'))
        torch.testing.assert_close(pool_context_logits(varied, 'softor', alpha=0), varied[0])

    def test_softor_rewards_partial_support_and_has_gradients(self):
        x = torch.tensor([[0.], [-2.], [2.]], requires_grad=True)
        mean = pool_context_logits(x, 'mean', alpha=1)
        softor = pool_context_logits(x, 'softor', alpha=1)
        self.assertGreater(float(softor), float(mean))
        self.assertLess(float(softor), 2.)
        softor.sum().backward()
        self.assertTrue(bool((x.grad[1:] > 0).all()))

    def test_diagnosis_and_random_groups_match_sizes_and_cell_ids(self):
        cells = ['ACH-%06d' % i for i in range(80)]
        mapping = {c: 'A' if i < 30 else 'B' if i < 60 else None for i, c in enumerate(cells)}
        names, real, random = make_groups(cells, mapping, n_named=2)
        self.assertEqual(names, ['A', 'B', 'remaining_cell_lines'])
        for group in range(3):
            self.assertEqual(sum(v == group for v in real.values()), sum(v == group for v in random.values()))
        _, other_real, other_random = make_groups(cells[::-1], mapping, n_named=2)
        self.assertEqual(real, other_real)
        self.assertEqual(random, other_random)
        self.assertNotEqual(real, random)

    def test_context_scalers_do_not_fit_remaining_nodes(self):
        rng = np.random.default_rng(0)
        raw = rng.normal(size=(8, 10))
        params = dict(input_mean=np.zeros(10), input_scale=np.ones(10), pca_mean=np.zeros(10),
                      pca_components=rng.normal(size=(3, 10)), component_mean=np.zeros(3),
                      component_scale=np.ones(3), fit_ids=np.arange(5))
        cells = list(range(10))
        assigned = {c: c % 2 for c in cells}
        first, audit = context_features(raw, params, cells, assigned, 2)
        raw[5:] += 1000
        second, other_audit = context_features(raw, params, cells, assigned, 2)
        np.testing.assert_array_equal(first[:, :5], second[:, :5])
        self.assertEqual(audit, other_audit)
        self.assertFalse(np.array_equal(first[:, 5:], second[:, 5:]))

    def test_batched_heterogeneous_graph_matches_independent_graphs(self):
        torch.manual_seed(0)
        data = HeteroData()
        data['lncRNA'].x = torch.randn(5, 4)
        data['drug'].x = torch.randn(3, 6)
        edge = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 1, 0]])
        data['lncRNA', 'LncDrug', 'drug'].edge_index = edge
        data['drug', 'rev_LncDrug', 'lncRNA'].edge_index = edge.flip(0)
        pairs = {'train': torch.tensor([[0, 2, 4], [2, 1, 0]])}
        weights = {k: torch.ones(e.shape[1]) for k, e in data.edge_index_dict.items()}
        model = ExperimentModel(data, 'LncDrug')
        views = {'lncRNA': torch.stack([data['lncRNA'].x * .7, data['lncRNA'].x + .3])}
        x, edges, indices, repeated, bags = pack_context_graph(data, pairs, weights, views, 'lncRNA')
        separate = []
        for expression in [data['lncRNA'].x, *views['lncRNA']]:
            current = dict(data.x_dict, lncRNA=expression)
            separate.append(model(current, data.edge_index_dict, pairs['train'], weights))
        expected = torch.stack(separate)
        actual = model(x, edges, indices['train'], repeated).view(bags, -1)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        model.zero_grad()
        expected.square().mean().backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
        model.zero_grad()
        actual.square().mean().backward()
        for name, p in model.named_parameters():
            torch.testing.assert_close(p.grad, gradients[name], atol=3e-6, rtol=3e-5)
        # Source and destination nodes always belong to the SAME graph copy.
        for key, index in edges.items():
            torch.testing.assert_close(index[0] // data[key[0]].num_nodes,
                                       index[1] // data[key[2]].num_nodes)


if __name__ == '__main__':
    unittest.main()
