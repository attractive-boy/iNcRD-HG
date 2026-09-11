"""Control equivalence and the mathematical contracts of attention smoothing."""
import copy
import unittest

import torch

from improve.drug_condition_attention import make_base, masked_weights, set_seed
from improve.edge_path_attention import configure
from improve.edge_path_modules import EdgePathModel
from improve.smooth_path_attention import ARMS, SmoothPathModel, smooth_weights
from improve.test_edge_path_attention import fixture


class SmoothingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()

    def test_smoothing_normalises_and_keeps_empty_or_singleton_rows(self):
        scores = torch.tensor([[2., -1., 8.], [2., 3., 9.], [7., 1., 2.]])
        valid = torch.tensor([[True, True, False], [False, False, False], [False, True, False]])
        for temperature, fraction in ((1., 1.), (4., 1.), (1., .25), (1., 0.)):
            weights = smooth_weights(scores, valid, temperature, fraction)
            torch.testing.assert_close(weights.sum(1), torch.tensor([1., 0., 1.]))
            self.assertTrue(torch.equal(weights[~valid], torch.zeros_like(weights[~valid])))
            self.assertEqual(weights[2, 1].item(), 1.)

    def test_temperature_increases_entropy_and_shrinkage_has_exact_distance(self):
        scores = torch.tensor([[3., 1., -2.], [-1., 2., 0.]])
        valid = torch.ones_like(scores, dtype=torch.bool)
        original = masked_weights(scores, valid)
        hot = smooth_weights(scores, valid, 4.)
        entropy = lambda weights: -(weights * weights.log()).sum(1)
        self.assertTrue(bool((entropy(hot) > entropy(original)).all()))
        uniform = torch.ones_like(scores) / 3
        shrunk = smooth_weights(scores, valid, fraction=.25)
        torch.testing.assert_close(shrunk - uniform, .25 * (original - uniform))

    def test_common_models_have_identical_values_gradients_and_updates(self):
        data, key, assets, _ = fixture()
        config = dict(hidden=8, rna='miRNA', aggr='mean')
        for arm in ('base', 'path_uniform', 'path_attention'):
            base = make_base(data, config, 42, 'cpu')
            set_seed(42 + 3000003)
            original = EdgePathModel(copy.deepcopy(base), data, assets, config, arm, 42, width=4)
            set_seed(42 + 3000003)
            current = SmoothPathModel(copy.deepcopy(base), assets, config, arm, width=4)
            optimizers = [torch.optim.RMSprop(model.parameters(), lr=.001) for model in (original, current)]
            for _ in range(4):
                predictions = []
                for model, optimizer in zip((original, current), optimizers):
                    optimizer.zero_grad()
                    scores = model(data.x_dict, data.edge_index_dict, 'train')
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, data[key].edge_label)
                    loss.backward()
                    optimizer.step()
                    predictions.append(scores.detach())
                torch.testing.assert_close(*predictions, rtol=0, atol=0)
            for name, value in original.state_dict().items():
                torch.testing.assert_close(value, current.state_dict()[name], rtol=0, atol=0)

    def test_new_attention_parameters_receive_finite_gradients(self):
        data, key, assets, _ = fixture()
        config = dict(hidden=8, rna='miRNA', aggr='mean')
        for arm in ('temperature4', 'shrink25'):
            base = make_base(data, config, 42, 'cpu')
            model = SmoothPathModel(base, assets, config, arm, width=4)
            torch.nn.init.normal_(model.branch.head[-1].weight, std=.1)
            scores = model(data.x_dict, data.edge_index_dict, 'train')
            torch.nn.functional.binary_cross_entropy_with_logits(scores, data[key].edge_label).backward()
            for parameter in (model.branch.query_r.weight, model.branch.query_d.weight, model.branch.keys[0].weight):
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                self.assertGreater(float(parameter.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
