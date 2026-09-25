import unittest

import numpy as np
import torch
from torch_geometric.data import HeteroData

from improve.pair_exchange_modules import remove_target_pairs, sample_queries
from improve.ragt_modules import StructuralHGT


def graph_fixture():
    torch.manual_seed(3)
    g = HeteroData()
    for nt, n in [('lncRNA', 8), ('miRNA', 5), ('drug', 6), ('circRNA', 4)]:
        g[nt].x = torch.randn(n, 7)
    for src, rel, dst, edges in [
            ('lncRNA', 'LncLnc', 'lncRNA', [[0,1,2,3,4,5,6,7],[1,2,3,4,5,6,7,0]]),
            ('miRNA', 'MiMi', 'miRNA', [[0,1,2,3,4],[1,2,3,4,0]]),
            ('drug', 'DrugDrug', 'drug', [[0,1,2,3,4,5],[1,2,3,4,5,0]]),
            ('lncRNA', 'LncMi', 'miRNA', [[0,1,2,3,4],[0,1,2,3,4]]),
            ('miRNA', 'MiDrug', 'drug', [[0,1,2,3,4],[0,1,2,3,4]]),
            ('circRNA', 'CircCirc', 'circRNA', [[0,1,2,3],[1,2,3,0]]),
            ('circRNA', 'CircDrug', 'drug', [[0,1,2,3],[0,1,2,3]])]:
        g[src, rel, dst].edge_index = torch.tensor(edges)
    pairs = np.array([[i for i in range(8) for _ in range(3)],
                      [j for i in range(8) for j in range(3)]])
    labels = np.array([1, 0, 0]*8, dtype=np.float32)
    pos = torch.from_numpy(pairs[:, labels == 1])
    g['lncRNA', 'LncDrug', 'drug'].edge_index = pos
    g['drug', 'rev_LncDrug', 'lncRNA'].edge_index = pos.flip(0)
    return g, pairs, labels


class RAGTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def _model(self, **kwargs):
        g, _, _ = graph_fixture()
        defaults = dict(width=16, depth=1, heads=4, dropout=.0, fusion='add', gate_init=-2.)
        defaults.update(kwargs)
        return g, StructuralHGT({nt: x.shape[1] for nt, x in g.x_dict.items()},
                                list(g.edge_types), 'lncRNA', 6, **defaults)

    def test_forward_shapes_and_endpoint_readout(self):
        g, pairs, y = graph_fixture()
        _, model = self._model()
        features = torch.randn(pairs.shape[1], 6)
        logits = model(g.x_dict, g.edge_index_dict,
                       torch.from_numpy(pairs), features)
        self.assertEqual(logits.shape, (pairs.shape[1],))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        z = model.encode(g.x_dict, g.edge_index_dict)
        self.assertEqual(z['lncRNA'].shape, (g['lncRNA'].num_nodes, model.width))
        self.assertEqual(z['drug'].shape, (g['drug'].num_nodes, model.width))

    def test_concatenation_fusion_matches_manual_input(self):
        g, pairs, y = graph_fixture()
        _, model = self._model(fusion='concat')
        features = torch.randn(pairs.shape[1], 6)
        z = model.encode(g.x_dict, g.edge_index_dict)
        manual = model.decoder(torch.cat([
            z['lncRNA'][pairs[0]], z['drug'][pairs[1]],
            (z['lncRNA'][pairs[0]]-z['drug'][pairs[1]]).abs(),
            z['lncRNA'][pairs[0]]*z['drug'][pairs[1]], features], -1)).view(-1)
        torch.testing.assert_close(model.decode(z, torch.from_numpy(pairs), features), manual)

    def test_additive_gate_starts_at_fixed_small_value(self):
        _, model = self._model()
        self.assertAlmostEqual(float(model.graph_gate.sigmoid()), 0.119203, places=5)

    def test_query_positive_never_visible_to_its_own_forward(self):
        g, pairs, y = graph_fixture()
        x_dict = {nt: g[nt].x for nt in g.node_types}
        edge_index_dict = {k: g[k].edge_index for k in g.edge_types}
        weights = {k: torch.ones(g[k].edge_index.size(1)) for k in g.edge_types}
        rna, n_drug = 'lncRNA', g['drug'].num_nodes
        query = sample_queries(y, .5, 1000003)
        held = pairs[:, query[y[query] == 1]]
        edges, w = remove_target_pairs(edge_index_dict, weights, 'LncDrug', rna, n_drug, held)
        for key in (('lncRNA', 'LncDrug', 'drug'), ('drug', 'rev_LncDrug', 'lncRNA')):
            direction = edges[key] if key[0] == 'lncRNA' else edges[key].flip(0)
            present = set(map(tuple, direction.numpy().T))
            self.assertFalse(present & set(map(tuple, held.T)))

    def test_training_gradients_are_finite(self):
        g, pairs, y = graph_fixture()
        _, model = self._model()
        features = torch.randn(pairs.shape[1], 6)
        model.train()
        model(g.x_dict, g.edge_index_dict, torch.from_numpy(pairs), features).sum().backward()
        for param in model.parameters():
            if param.grad is not None:
                self.assertTrue(bool(torch.isfinite(param.grad).all()))

    def test_degree_encoding_is_order_stable(self):
        g, pairs, y = graph_fixture()
        _, model = self._model()
        a = model._degrees(g.x_dict, g.edge_index_dict)
        b = model._degrees(g.x_dict, g.edge_index_dict)
        for nt in g.node_types:
            np.testing.assert_array_equal(a[nt], b[nt])
            self.assertEqual(a[nt].shape, (g[nt].num_nodes, len(g.edge_types)))


if __name__ == '__main__':
    unittest.main()