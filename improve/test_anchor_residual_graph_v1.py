"""Residual likelihood, raw-logit expectation, and heldout fitting boundary."""
from copy import deepcopy
import unittest

import numpy as np
import torch

from improve import anchor_residual_graph_v1 as experiment
from improve.test_nested_anchor_v2 import fixture


def parameters(family):
    parent = experiment.nested.base.read_json(experiment.ROOT / 'results' /
        ('residual_' + family + '_20260915/protocol.json'))
    p = deepcopy(parent)
    p.update(smoke=True, private_rng_offset=29015)
    p['core_parameters'].update(width=8, heads=2, depth=1, communities=4,
                                 posterior_samples=3, expander_degree=2)
    return p


class GraphResidualTests(unittest.TestCase):
    def test_prediction_is_mean_raw_residual(self):
        class Constant(torch.nn.Module):
            def forward(self, xs, graph, pairs):
                return pairs[0].float() + graph
        pairs = torch.tensor([[1, 3, 7], [2, 4, 8]])
        value = experiment.predict(Constant(), [], [-8., 12.], pairs, dict(family='geometry'))
        torch.testing.assert_close(value, torch.tensor([3., 5., 9.], dtype=torch.float64))
        a = dict(y_val=np.array([0., 1., 0.]), pairs_val=pairs.numpy())
        for name in ('anchor', 'tabicl', 'gnn'):
            a[name + '_666'] = np.array([-5., 2., 9.])
        z = experiment.score_arrays(a, np.zeros(3), 666)
        np.testing.assert_array_equal(z['anchor_composed'], z['anchor'])
        np.testing.assert_array_equal(z['offset_composed'], z['anchor'])

    def test_fixed_training_rejects_query_data_and_freezes_marginals(self):
        a, _ = fixture()
        a['anchor_oof'] = np.linspace(-1, 1, len(a['y_train']))
        p = parameters('geometry')
        contaminated = dict(a, y_val=np.array([1.]))
        with self.assertRaisesRegex(ValueError, 'training inputs only'):
            experiment.fit(contaminated, p, 'attention_only', .001, 3, 'cpu', fixed_epochs=2)
        model, prediction, trace, detail = experiment.fit(a, p, 'attention_only', .001, 3, 'cpu', fixed_epochs=2)
        self.assertIsNone(prediction)
        self.assertTrue(detail['fixed_step_training'])
        self.assertNotIn('anchor_composed', trace[-1])
        for head in (model.rna_degree, model.drug_degree):
            for value in head.parameters():
                self.assertFalse(value.requires_grad)
                self.assertEqual(value.detach().abs().sum(), 0)

    def test_posterior_residual_replay_is_query_order_invariant(self):
        a, _ = fixture()
        a['anchor_oof'] = np.zeros(len(a['y_train']))
        p = parameters('block')
        torch.manual_seed(10)
        model, xs, graphs, _ = experiment.setup(a, p, 'full', 'cpu')
        pairs = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None])
        direct = experiment.predict(model, xs, graphs, pairs, p)
        reverse = experiment.predict(model, xs, graphs, pairs.flip(1), p).flip(0)
        chunks = torch.cat([experiment.predict(model, xs, graphs, pairs[:, i:i + 7], p)
                            for i in range(0, pairs.shape[1], 7)])
        torch.testing.assert_close(direct, reverse, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(direct, chunks, atol=3e-5, rtol=3e-5)


if __name__ == '__main__':
    unittest.main()
