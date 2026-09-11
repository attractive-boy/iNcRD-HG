"""Leakage guards, fallback behavior, matched experts and routing gradients."""
import copy
import unittest

import numpy as np
import torch

from improve.drug_condition_attention import make_base, set_seed
from improve.edge_path_attention import configure
from improve.smooth_path_attention import SmoothPathModel
from improve.complementary_attention_modules import ARMS, ComplementaryModel, routing_assets
from improve.test_edge_path_attention import fixture


class ComplementaryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()

    def test_query_degree_excludes_candidate_and_ignores_labels(self):
        data, key, paths, _ = fixture()
        first = routing_assets(data, paths, 'miRNA', 'MiDrug')
        changed = copy.deepcopy(data)
        changed[key].edge_index = changed[key].edge_index[:, 1:]
        second = routing_assets(changed, paths, 'miRNA', 'MiDrug')
        # Row 0 queries the removed (0, 0) edge: its degrees must not change.
        torch.testing.assert_close(first['train']['features'][0], second['train']['features'][0], rtol=0, atol=0)
        changed[key].edge_label = 1 - changed[key].edge_label
        third = routing_assets(changed, paths, 'miRNA', 'MiDrug')
        for fold in first:
            torch.testing.assert_close(second[fold]['features'], third[fold]['features'], rtol=0, atol=0)
            self.assertFalse(bool(first[fold]['available'][0, 2]))

    def test_all_arms_start_at_the_base_and_share_expert_initializations(self):
        data, key, paths, _ = fixture()
        route = routing_assets(data, paths, 'miRNA', 'MiDrug')
        config = dict(hidden=8, rna='miRNA', aggr='mean')
        base = make_base(data, config, 42, 'cpu')
        expected = base(data.x_dict, data.edge_index_dict, paths['pairs']['train']).detach()
        states = {}
        for arm in ARMS:
            model = ComplementaryModel(copy.deepcopy(base), paths, route, config, arm, 42, width=4)
            actual = model(data.x_dict, data.edge_index_dict, 'train')
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if model.channel is not None:
                states[arm, 'channel'] = model.channel.state_dict()
            if model.path is not None:
                states[arm, 'path'] = model.path.state_dict()
        for expert, reference in (('channel', 'channel_attention'), ('path', 'path_attention')):
            for arm in ('fusion_fixed', 'fusion_global', 'fusion_gate', 'fusion_no_null'):
                for name, value in states[reference, expert].items():
                    torch.testing.assert_close(value, states[arm, expert][name], rtol=0, atol=0)

    def test_isolated_path_matches_previous_model_predictions_and_updates(self):
        data, key, paths, _ = fixture()
        route = routing_assets(data, paths, 'miRNA', 'MiDrug')
        config = dict(hidden=8, rna='miRNA', aggr='mean')
        base = make_base(data, config, 42, 'cpu')
        # FX GraphModule deepcopy can consume RNG; materialize both base copies
        # before resetting the expert seed, as the real trainer does.
        original_base, current_base = copy.deepcopy(base), copy.deepcopy(base)
        set_seed(42 + 3000003)
        original = SmoothPathModel(original_base, paths, config, 'temperature4', 4)
        current = ComplementaryModel(current_base, paths, route, config, 'path_attention', 42, width=4)
        optimizers = [torch.optim.RMSprop(m.parameters(), lr=.001) for m in (original, current)]
        for _ in range(4):
            outputs = []
            for model, optimizer in zip((original, current), optimizers):
                optimizer.zero_grad()
                scores = model(data.x_dict, data.edge_index_dict, 'train')
                torch.nn.functional.binary_cross_entropy_with_logits(scores, data[key].edge_label).backward()
                optimizer.step()
                outputs.append(scores.detach())
            torch.testing.assert_close(*outputs, rtol=0, atol=0)
        for name, value in original.branch.state_dict().items():
            torch.testing.assert_close(value, current.path.state_dict()[name], rtol=0, atol=0)

    def test_missing_path_is_masked_and_null_can_recover_base(self):
        data, key, paths, _ = fixture()
        route = routing_assets(data, paths, 'miRNA', 'MiDrug')
        config = dict(hidden=8, rna='miRNA', aggr='mean')
        base = make_base(data, config, 42, 'cpu')
        model = ComplementaryModel(base, paths, route, config, 'fusion_gate', 42, width=4)
        with torch.no_grad():
            model.path.head[-1].bias.fill_(3.)
            model.router.keys.bias.copy_(torch.tensor([1000., -1000., -1000.]))
        actual = model(data.x_dict, data.edge_index_dict, 'train')
        expected = model(data.x_dict, data.edge_index_dict, 'train', 'base_only')
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with torch.no_grad():
            model.router.keys.bias.zero_()
        model(data.x_dict, data.edge_index_dict, 'train')
        weights = model.attention()['router']
        np.testing.assert_allclose(weights.sum(1), 1.)
        self.assertEqual(weights[0, 2], 0.)

    def test_router_and_both_experts_receive_finite_gradients(self):
        data, key, paths, _ = fixture()
        route = routing_assets(data, paths, 'miRNA', 'MiDrug')
        config = dict(hidden=8, rna='miRNA', aggr='mean')
        for arm in ('fusion_gate', 'fusion_gate_simple', 'fusion_no_null'):
            model = ComplementaryModel(make_base(data, config, 42, 'cpu'), paths, route, config, arm, 42, width=4)
            optimizer = torch.optim.RMSprop(model.parameters(), lr=.003)
            before = {name: value.clone() for name, value in model.named_parameters()}
            for _ in range(6):
                optimizer.zero_grad()
                scores = model(data.x_dict, data.edge_index_dict, 'train')
                loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, data[key].edge_label)
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                optimizer.step()
            changed = [name for name, value in model.named_parameters() if not torch.equal(before[name], value)]
            for prefix in ('channel.', 'path.', 'router.'):
                self.assertTrue(any(name.startswith(prefix) for name in changed), (arm, prefix))


if __name__ == '__main__':
    unittest.main()
