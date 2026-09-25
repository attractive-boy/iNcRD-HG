"""The new pair core under the existing nested-reference fitting contract."""
import unittest

import numpy as np
import torch

from improve import anchor_residual_graph_v2 as executor
from improve import triangular_pair_graph_v3 as core
from improve.test_nested_anchor_v2 import fixture


class PairExecutorTests(unittest.TestCase):
    def test_both_mechanism_arms_fit_with_training_inputs_only(self):
        a, _ = fixture()
        a['anchor_oof'] = np.linspace(-1., 1., len(a['y_train']))
        config = core.parameters(); config.update(width=8, heads=2, dropout=0., query_chunk=4)
        p = dict(smoke=True, family='triangular_pair', implementation_module='triangular_pair_graph_v3',
            source_sha256={'triangular_pair_graph_v3.py': executor.common.base.digest(core.__file__)},
            core_parameters=config, kinds=list(core.KINDS), epochs=3, check_interval=1,
            weight_decay=.001, private_rng_offset=29015, strong_weight=.1)
        for kind in core.KINDS:
            model, values, trace, detail = executor.fit(a, p, kind, .003, 492, 'cpu', fixed_epochs=3)
            self.assertIsNone(values)
            self.assertTrue(detail['fixed_step_training'])
            self.assertEqual(len(trace), 3)
            self.assertTrue(all(np.isfinite(t['loss']) for t in trace))
            self.assertTrue(all('anchor_composed' not in t for t in trace))
            self.assertGreater(model.readout[-1].weight.abs().sum().item(), 0.)
        with self.assertRaisesRegex(ValueError, 'training inputs only'):
            executor.fit(dict(a, y_val=np.ones(4)), p, 'full', .003, 492, 'cpu', fixed_epochs=3)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
