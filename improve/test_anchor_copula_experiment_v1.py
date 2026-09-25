"""Explicit query reference, unchanged null, and training-only fit contract."""
import unittest

import numpy as np
import torch

from improve import anchor_copula_experiment_v1 as experiment
from improve import bernoulli_copula_graph_v1 as copula
from improve.test_nested_anchor_v2 import fixture


def parameters():
    core = copula.parameters(); core.update(width=8, rank=3, depth=1, dropout=0.)
    return dict(smoke=True, family='bernoulli_copula', implementation_module='bernoulli_copula_graph_v1',
        source_sha256={'bernoulli_copula_graph_v1.py': experiment.common.base.digest(copula.__file__)},
        core_parameters=core, kinds=list(copula.KINDS), epochs=2, check_interval=1,
        weight_decay=.001, private_rng_offset=29015, strong_weight=.1)


class CopulaExecutorTests(unittest.TestCase):
    def test_training_only_and_source_identity(self):
        a, _ = fixture()
        a['anchor_oof'] = np.linspace(-1., 1., len(a['y_train']))
        p = parameters()
        with self.assertRaisesRegex(ValueError, 'training inputs only'):
            experiment.fit(dict(a, y_val=np.ones(2)), p, 'full', .001, 3, 'cpu', fixed_epochs=2)
        model, prediction, trace, detail = experiment.fit(a, p, 'full', .001, 3, 'cpu', fixed_epochs=2)
        self.assertIsNone(prediction)
        self.assertTrue(detail['conditional_query_marginal_explicit'])
        self.assertFalse(detail['support_marginals_use_global_oof'])
        self.assertNotIn('anchor_composed', trace[-1])
        self.assertTrue(all(np.isfinite(row['loss']) for row in trace))
        p['source_sha256']['bernoulli_copula_graph_v1.py'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'implementation changed'):
            experiment.library(p)

    def test_reference_follows_query_order_and_zero_residual_recovers_anchor(self):
        class Inspect(torch.nn.Module):
            def forward(self, xs, graph, pairs, query_logits):
                return pairs[0].double() + query_logits + graph
        pairs = torch.tensor([[1, 3, 7], [2, 4, 8]])
        offsets = torch.tensor([-10., 20., 30.], dtype=torch.float64)
        value = experiment.predict(Inspect(), [], [-2., 2.], pairs, {}, offsets)
        torch.testing.assert_close(value, pairs[0] + offsets)
        reversed_value = experiment.predict(Inspect(), [], [-2., 2.], pairs.flip(1), {}, offsets.flip(0))
        torch.testing.assert_close(value, reversed_value.flip(0))
        a = dict(y_val=np.array([0., 1., 0.]), pairs_val=pairs.numpy())
        for name in ('anchor', 'gnn', 'tabicl'):
            a[name + '_666'] = np.array([-5., 2., 9.])
        z = experiment.score_arrays(a, np.zeros(3), 666)
        np.testing.assert_array_equal(z['anchor_composed'], z['anchor'])
        np.testing.assert_array_equal(z['offset_composed'], z['anchor'])


if __name__ == '__main__':
    unittest.main()
