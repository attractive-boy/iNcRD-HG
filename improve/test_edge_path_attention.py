"""Path validity, edge-weight controls and reproducibility contracts."""
import copy
import unittest

import numpy as np
import torch

from improve.edge_path_attention import configure
from improve.edge_path_modules import ARMS, AttributeConv, EdgePathModel, PathBranch, build_assets, shuffle_attributes
from improve.drug_condition_attention import make_base, masked_weights, set_seed
from improve.attention_mechanisms import parameter_hash
from improve.test_attention_mechanisms import tiny_data
from torch_geometric.nn import GeneralConv


def fixture():
    data, key = tiny_data()
    # Two real intermediate RNAs connect RNA 0 to drug 1.
    data[key].edge_index = torch.tensor([[0, 1, 2], [0, 1, 1]])
    data['drug', 'rev_MiDrug', 'miRNA'].edge_index = data[key].edge_index.flip(0)
    data[key].edge_label_index = torch.tensor([[0, 0, 1, 2, 3], [0, 1, 1, 1, 2]])
    data[key].edge_label = torch.tensor([1., 0., 1., 1., 0.])
    # Explicit self edges must never create r -> r -> candidate-drug paths.
    relation = ('miRNA', 'MiMi', 'miRNA')
    data[relation].edge_index = torch.cat([data[relation].edge_index, torch.arange(4).repeat(2, 1)], 1)
    matrix = np.full((4, 4), .6, dtype=np.float32)
    np.fill_diagonal(matrix, 1.)
    pairs = {fold: data[key].edge_label_index.clone() for fold in ('train', 'val', 'test')}
    assets = build_assets(data, pairs, {relation: matrix}, 'miRNA')
    return data, key, assets, matrix


class EdgePathChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()

    def test_paths_match_enumeration_and_exclude_candidate_edge_shortcut(self):
        data, key, assets, matrix = fixture()
        self.assertEqual(len(assets['families']), 1)
        pack = assets['packs']['train'][0]
        self.assertEqual(pack['counts'][0].item(), 0.)
        mids = pack['ids'][1][pack['valid'][1]].tolist()
        self.assertEqual(mids, [1, 2])
        first = set(map(tuple, data['miRNA', 'MiMi', 'miRNA'].edge_index.t().tolist()))
        second = set(map(tuple, data[key].edge_index.t().tolist()))
        for row, (rna, drug) in enumerate(assets['pairs']['train'].t().tolist()):
            expected = [m for m in range(4) if m != rna and (rna, m) in first and (m, drug) in second]
            actual = pack['ids'][row][pack['valid'][row]].tolist()
            self.assertEqual(actual, expected)
        changed = copy.deepcopy(data)
        changed[key].edge_index = changed[key].edge_index[:, 1:]
        rebuilt = build_assets(changed, assets['pairs'], {('miRNA', 'MiMi', 'miRNA'): matrix}, 'miRNA')
        # Removing candidate (0, 0) cannot alter its own path features.
        self.assertEqual(rebuilt['packs']['train'][0]['counts'][0].item(), pack['counts'][0].item())
        changed[key].edge_label = 1 - changed[key].edge_label
        ignored_labels = build_assets(changed, assets['pairs'], {('miRNA', 'MiMi', 'miRNA'): matrix}, 'miRNA')
        for name in ('ids', 'valid', 'counts'):
            self.assertTrue(torch.equal(rebuilt['packs']['train'][0][name], ignored_labels['packs']['train'][0][name]))

    def test_attribute_shuffle_preserves_self_edges_distribution_and_reciprocity(self):
        edges = torch.tensor([[0, 1, 0, 2, 1, 2, 0, 1, 2], [1, 0, 2, 0, 2, 1, 0, 1, 2]])
        signed = torch.tensor([.4, .4, -.5, -.5, .8, .8, 1., 1., 1.])
        attributes = torch.stack([signed, signed.abs()], -1)
        shuffled = shuffle_attributes(attributes, edges, 33)
        self.assertTrue(torch.equal(shuffled[0], shuffled[1]))
        self.assertTrue(torch.equal(shuffled[2], shuffled[3]))
        self.assertTrue(torch.equal(shuffled[4], shuffled[5]))
        self.assertTrue(torch.equal(shuffled[6:], attributes[6:]))
        torch.testing.assert_close(shuffled[:6, 0].sort().values, attributes[:6, 0].sort().values)
        self.assertTrue(torch.equal(shuffled, shuffle_attributes(attributes, edges, 33)))

    def test_edge_control_preserves_self_loops_total_weight_and_initial_base(self):
        set_seed(4)
        original = GeneralConv((5, 5), 8)
        edges = torch.tensor([[0, 1, 2, 0, 1, 2], [0, 0, 0, 0, 1, 2]])
        signed = torch.tensor([1., -.5, .8, 1., 1., 1.])
        attributes = torch.stack([signed, signed.abs()], -1)
        x = torch.randn(3, 5)
        expected = original((x, x), edges)
        for arm in ('edge_blind', 'edge_attributes', 'edge_shuffled', 'edge_fixed'):
            module = AttributeConv(original, edges, attributes, 3, arm, 33, 4)
            actual = module((x, x), edges)
            w = module.last_weights
            self.assertTrue(torch.equal(w[edges[0] == edges[1]], torch.ones(4)))
            total = torch.zeros(3).index_add(0, edges[1], w)
            torch.testing.assert_close(total, torch.tensor([4., 1., 1.]))
            if arm != 'edge_fixed':
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_path_weights_are_normalised_and_empty_paths_stay_zero(self):
        data, key, assets, _ = fixture()
        cfg = dict(hidden=8, rna='miRNA', aggr='mean')
        for arm in ('path_uniform', 'path_attention'):
            base = make_base(data, cfg, 42, 'cpu')
            model = EdgePathModel(base, data, assets, cfg, arm, 42, width=4, edge_width=4)
            model(data.x_dict, data.edge_index_dict, 'train')
            weights = model.branch.last_weights['path_0']
            counts = assets['packs']['train'][0]['counts']
            torch.testing.assert_close(weights.sum(-1), (counts > 0).float())
            self.assertEqual(float(weights[0].abs().sum()), 0.)

    def test_sparse_path_evaluation_matches_dense_reference_values_and_gradients(self):
        data, key, assets, _ = fixture()
        cfg = dict(hidden=8, rna='miRNA', aggr='mean')
        base = make_base(data, cfg, 42, 'cpu')
        z = {nt: x.detach() for nt, x in base.encoder(data.x_dict, data.edge_index_dict).items()}
        sparse = PathBranch(8, 'miRNA', assets, 'path_attention', width=4)
        torch.nn.init.normal_(sparse.head[-1].weight, std=.1)
        dense = copy.deepcopy(sparse)
        actual = sparse(z, assets['pairs']['train'], 'train')
        normalised = {nt: torch.nn.functional.layer_norm(x, (x.size(-1),)) for nt, x in z.items()}
        r, d = assets['pairs']['train']
        query = dense.query_r(normalised['miRNA'][r]) + dense.query_d(normalised['drug'][d])
        parts = []
        for i, family in enumerate(dense.families):
            pack = assets['packs']['train'][i]
            middle = normalised[family['middle']]
            keys = dense.keys[i](middle)[pack['ids']]
            scores = (query[:, None, :] * keys).sum(-1) / 2.
            weights = masked_weights(scores, pack['valid'])
            values = dense.values[i](middle)[pack['ids']]
            parts.extend([(weights[:, :, None] * values).sum(1), pack['counts'].log1p()[:, None]])
        expected = dense.head(torch.cat(parts, -1)).view(-1)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        for a, b in zip(sparse.parameters(), dense.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-6)

    def test_all_arms_train_and_repeat_exactly(self):
        data, key, assets, _ = fixture()
        cfg = dict(hidden=8, rna='miRNA', aggr='mean')
        for arm in ARMS:
            with self.subTest(arm=arm):
                states = []
                for repeat in range(2):
                    base = make_base(data, cfg, 42, 'cpu')
                    expected = base(data.x_dict, data.edge_index_dict, data[key].edge_label_index).detach()
                    set_seed(42 + 3000003)
                    model = EdgePathModel(base, data, assets, cfg, arm, 42, width=4, edge_width=4)
                    initial = model(data.x_dict, data.edge_index_dict, 'train')
                    if arm != 'edge_fixed':
                        torch.testing.assert_close(initial, expected, rtol=0, atol=0)
                    set_seed(42 + 9000007)
                    before = {n: p.clone() for n, p in model.named_parameters()}
                    optimizer = torch.optim.RMSprop(model.parameters(), lr=.001, alpha=.99)
                    for _ in range(5):
                        optimizer.zero_grad()
                        scores = model(data.x_dict, data.edge_index_dict, 'train')
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, data[key].edge_label)
                        loss.backward()
                        for p in model.parameters():
                            if p.grad is not None:
                                self.assertTrue(torch.isfinite(p.grad).all())
                        optimizer.step()
                    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
                    self.assertTrue(changed)
                    if arm == 'edge_attributes':
                        self.assertTrue(any(n.endswith('scorer.weight') for n in changed))
                    if arm == 'path_attention':
                        self.assertTrue(any('query_' in n for n in changed))
                    normal = model(data.x_dict, data.edge_index_dict, 'train')
                    if arm in ('edge_attributes', 'path_attention'):
                        altered = model(data.x_dict, data.edge_index_dict, 'train', 'uniform')
                        self.assertTrue(torch.isfinite(altered).all())
                        replay = model(data.x_dict, data.edge_index_dict, 'train')
                        torch.testing.assert_close(normal, replay, rtol=0, atol=0)
                    states.append(parameter_hash(model))
                self.assertEqual(states[0], states[1])


if __name__ == '__main__':
    unittest.main()
