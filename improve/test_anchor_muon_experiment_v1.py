"""Exact ordinary-AdamW trajectory for possible baseline weight adoption."""
import unittest

import numpy as np
import torch

from improve import anchor_muon_experiment_v1 as muon
from improve import anchor_sharpness_experiment_v1 as friendly
from improve import muon_graph_v1 as core
from improve import sharpness_graph_v1 as old_core
from improve import spectral_muon_v1 as spectral
from improve.test_nested_anchor_v2 import fixture


def protocols():
    parameters = core.parameters(); parameters.update(width=8, heads=2, depth=2, dropout=.1)
    p = dict(smoke=True, family='spectral_muon', implementation_module='muon_graph_v1',
        source_sha256={'muon_graph_v1.py': muon.common.base.digest(core.__file__)},
        core_parameters=parameters, kinds=list(core.KINDS), epochs=5, check_interval=5,
        weight_decay=.001, private_rng_offset=29015, strong_weight=.1,
        optimizer_parameters=spectral.parameters())
    previous = dict(p, family='friendly_sharpness', implementation_module='sharpness_graph_v1',
        source_sha256={'sharpness_graph_v1.py': muon.common.base.digest(old_core.__file__)},
        kinds=list(old_core.KINDS), optimizer_parameters=dict(rho=.05, sigma=1., decay=.9, clip=5.))
    return p, previous


class MuonExecutorTests(unittest.TestCase):
    def test_adamw_matches_previous_two_pass_erm_exactly(self):
        a, _ = fixture(); a['anchor_oof'] = np.linspace(-2, 2, len(a['y_train']))
        p, previous = protocols()
        model, _, _, detail = muon.fit(a, p, 'adamw', .001, 53, 'cpu', fixed_epochs=5)
        state = torch.get_rng_state().clone()
        old, _, _, old_detail = friendly.fit(a, previous, 'erm', .001, 53, 'cpu', fixed_epochs=5)
        self.assertEqual(detail['gradient_evaluations'], 5)
        self.assertEqual(old_detail['gradient_evaluations'], 10)
        torch.testing.assert_close(state, torch.get_rng_state(), atol=0, rtol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, old.state_dict()[key], atol=0, rtol=0)

    def test_boundary_and_identity_remain_enforced(self):
        a, _ = fixture(); a['anchor_oof'] = np.zeros_like(a['y_train'], dtype=np.float64)
        p, _ = protocols()
        with self.assertRaisesRegex(ValueError, 'training inputs only'):
            muon.fit(dict(a, y_val=np.ones(3)), p, 'full', .001, 1, 'cpu', fixed_epochs=1)
        with self.assertRaisesRegex(ValueError, 'cuda:1'):
            muon.fit(a, p, 'full', .001, 1, 'cuda:0', fixed_epochs=1)
        a['pair_offset'] = np.array([0, 14])
        model, _, trace, detail = muon.fit(a, p, 'full', .001, 1, 'cpu', fixed_epochs=0)
        self.assertFalse(detail['trained_neural_model'])
        self.assertEqual(detail['training_steps'], 0)
        self.assertFalse(model.state_dict())
        self.assertEqual(trace, [])


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
