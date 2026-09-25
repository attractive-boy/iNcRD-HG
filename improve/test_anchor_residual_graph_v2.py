"""Prospective module identity, fixed training boundary, and exact null."""
from copy import deepcopy
import unittest

import numpy as np
import torch

from improve import anchor_residual_graph_v2 as experiment
from improve import energy_graph_v1 as energy
from improve.test_nested_anchor_v2 import fixture


def parameters():
    core = energy.parameters()
    core.update(width=8, depth=2, dropout=0.)
    return dict(smoke=True, family='energy', implementation_module='energy_graph_v1',
        source_sha256={'energy_graph_v1.py': experiment.common.base.digest(energy.__file__)},
        core_parameters=core, kinds=list(energy.KINDS), epochs=2, check_interval=1,
        weight_decay=.001, private_rng_offset=29015, strong_weight=.1)


class ProspectiveResidualTests(unittest.TestCase):
    def test_fixed_training_rejects_query_outcomes_and_source_mismatch(self):
        a, _ = fixture()
        a['anchor_oof'] = np.linspace(-1., 1., len(a['y_train']))
        p = parameters()
        contaminated = dict(a, y_val=np.array([1.]))
        with self.assertRaisesRegex(ValueError, 'training inputs only'):
            experiment.fit(contaminated, p, 'full', .001, 3, 'cpu', fixed_epochs=2)
        wrong = deepcopy(p)
        wrong['source_sha256']['energy_graph_v1.py'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'implementation changed'):
            experiment.library(wrong)
        model, prediction, trace, detail = experiment.fit(a, p, 'full', .001, 3, 'cpu', fixed_epochs=2)
        self.assertIsNone(prediction)
        self.assertTrue(detail['fixed_step_training'])
        self.assertNotIn('anchor_composed', trace[-1])
        self.assertTrue(all(np.isfinite(row['loss']) for row in trace))

    def test_prediction_is_raw_residual_mean_and_null_is_exact_anchor(self):
        class Constant(torch.nn.Module):
            def forward(self, xs, graph, pairs):
                return pairs[0].float() + graph
        pairs = torch.tensor([[1, 3, 7], [2, 4, 8]])
        value = experiment.predict(Constant(), [], [-8., 12.], pairs, dict(family='energy'))
        torch.testing.assert_close(value, torch.tensor([3., 5., 9.], dtype=torch.float64))
        a = dict(y_val=np.array([0., 1., 0.]), pairs_val=pairs.numpy())
        for name in ('anchor', 'gnn', 'tabicl'):
            a[name + '_666'] = np.array([-5., 2., 9.])
        z = experiment.score_arrays(a, np.zeros(3), 666)
        np.testing.assert_array_equal(z['anchor_composed'], z['anchor'])
        np.testing.assert_array_equal(z['offset_composed'], z['anchor'])


if __name__ == '__main__':
    unittest.main()
