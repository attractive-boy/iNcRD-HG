"""Checks that the ablation changes the intended mechanism only."""
import unittest

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import GeneralConv

from improve.model import Model
from improve.network_feature_ablation import (ExperimentModel, WeightedGeneralConv,
                                               masked_graph)
from improve.network_feature_inputs import (crossfit_matrices, graph_weights,
                                            masked_pool)


class AblationContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)

    def test_padding_does_not_change_masked_pool(self):
        h = torch.randn(2, 3, 8)
        expected = masked_pool(h, torch.ones(2, 3))
        padded = torch.cat([h, torch.randn(2, 5, 8) * 100], dim=1)
        mask = torch.cat([torch.ones(2, 3), torch.zeros(2, 5)], dim=1)
        torch.testing.assert_close(masked_pool(padded, mask), expected)

    def test_scalar_weights_are_used_and_units_recover_generalconv(self):
        x = torch.randn(5, 3)
        edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        original = GeneralConv(3, 4)
        weighted = WeightedGeneralConv(3, 4)
        weighted.load_state_dict(original.state_dict())
        expected = original(x, edges)
        torch.testing.assert_close(weighted(x, edges, torch.ones(4)), expected)
        no_messages = weighted(x, edges, torch.zeros(4))
        torch.testing.assert_close(no_messages, weighted.lin_self(x))
        self.assertGreater(float((no_messages - expected).abs().max()), 1e-4)

    def test_heterogeneous_unit_weight_model_matches_original(self):
        data = HeteroData()
        data['lncRNA'].x = torch.randn(5, 3)
        data['drug'].x = torch.randn(4, 6)
        idx = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        data['lncRNA', 'LncDrug', 'drug'].edge_index = idx
        data['drug', 'rev_LncDrug', 'lncRNA'].edge_index = idx.flip(0)
        original = Model(data, 64, 'min', 'lncRNA')
        weighted = ExperimentModel(data, 'LncDrug')
        weights = {k: torch.ones(v.shape[1]) for k, v in data.edge_index_dict.items()}
        original(data.x_dict, data.edge_index_dict, idx)
        weighted(data.x_dict, data.edge_index_dict, idx, weights)
        weighted.load_state_dict(original.state_dict())
        torch.testing.assert_close(weighted(data.x_dict, data.edge_index_dict, idx, weights),
                                   original(data.x_dict, data.edge_index_dict, idx))

    def test_masks_align_with_weights_and_do_not_consume_global_rng(self):
        key = ('a', 'r', 'b')
        edges = {key: torch.stack([torch.arange(30), torch.arange(30)])}
        weights = {key: torch.arange(30, dtype=torch.float32)}
        before = torch.get_rng_state().clone()
        e, w = masked_graph(edges, weights, .2, 666)
        torch.testing.assert_close(torch.get_rng_state(), before)
        torch.testing.assert_close(e[key][0].float(), w[key])
        torch.rand(100)
        e2, w2 = masked_graph(edges, weights, .2, 666)
        torch.testing.assert_close(e[key], e2[key])
        torch.testing.assert_close(w[key], w2[key])

    def test_weighting_keeps_graph_and_incoming_mass(self):
        key = ('drug', 'DrugDrug', 'drug')
        data = HeteroData()
        data['drug'].x = torch.ones(3, 2)
        edges = torch.tensor([[0, 1, 2, 0], [1, 1, 1, 2]])
        data[key].edge_index = edges.clone()
        matrix = np.array([[1, .4, .8], [.4, 1, .6], [.8, .6, 1]], dtype=np.float32)
        weighted = graph_weights(data, {key: matrix}, True)[key]
        total = torch.zeros(3).index_add_(0, edges[1], weighted)
        count = torch.bincount(edges[1], minlength=3).float()
        torch.testing.assert_close(total, count)
        torch.testing.assert_close(data[key].edge_index, edges)
        self.assertFalse(torch.allclose(weighted, torch.ones_like(weighted)))

    def test_crossfit_removes_queried_positive_edges_without_mutating_source(self):
        pairs = np.array([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 2, 2, 3, 0, 3]])
        y = np.array([1, 0, 1, 0, 1, 0, 0, 1])
        A = np.zeros((4, 4))
        positive = pairs[:, y == 1]
        A[positive[0], positive[1]] = 1
        original = A.copy()
        seen = np.zeros(len(y), dtype=int)
        for take, Ak in crossfit_matrices(pairs, y, A):
            held = pairs[:, take & (y == 1)]
            self.assertFalse(Ak[held[0], held[1]].any())
            seen += take
        np.testing.assert_array_equal(seen, np.ones(len(y)))
        np.testing.assert_array_equal(A, original)


if __name__ == '__main__':
    unittest.main()
