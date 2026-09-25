"""Whole-training artifact equivalence and the newly matched zero-penalty control."""
import unittest

import numpy as np
import torch

from improve import anchor_ranking_experiment_v1 as old
from improve import anchor_ranking_experiment_v2 as revised
from improve import staco_ranking_v1 as old_core
from improve import staco_ranking_v2 as core
from improve.test_nested_anchor_v2 import fixture


def protocol(module, library):
    p = dict(library.parameters(), width=8, depth=1, dropout=.1, inner_steps=5,
             positive_batch=2, negative_batch=3)
    return dict(smoke=True, family='nonproximal_ranking', implementation_module=module,
        source_sha256={module+'.py': revised.common.base.digest(library.__file__)},
        core_parameters=p, kinds=list(library.KINDS), epochs=12, check_interval=5,
        weight_decay=.001, strong_weight=.1, private_rng_offset=29015)


class NonproximalRevisionChecks(unittest.TestCase):
    def test_full_training_replays_previous_no_proximal_weights_trace_state_and_rng(self):
        a, _ = fixture()
        a['anchor_oof'] = np.linspace(-1, 1, len(a['y_train']))
        before, _, trace_before, _ = old.fit(a, protocol('staco_ranking_v1', old_core),
            'no_proximal', .001, 468, 'cpu', fixed_epochs=12)
        rng = torch.get_rng_state().clone()
        after, _, trace_after, detail = revised.fit(a, protocol('staco_ranking_v2', core),
            'full', .001, 468, 'cpu', fixed_epochs=12)
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
        self.assertEqual(trace_before, trace_after)
        for key, value in before.state_dict().items():
            torch.testing.assert_close(value, after.state_dict()[key], atol=0, rtol=0)
        for key, value in before.training_risk_snapshot.items():
            actual = after.training_risk_snapshot[key]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, actual, atol=0, rtol=0)
            else:
                self.assertEqual(value, actual)
        self.assertEqual(detail['completed_averaging_stages'], 3)

    def test_new_bce_control_disables_both_proximal_terms_but_keeps_blocks_and_parameters(self):
        p = dict(core.parameters(), folds=1)
        labels = np.array([1, 1, 0, 0, 0, 0])
        full = core.TailState(labels, np.zeros(6, dtype=int), np.zeros(6), p, 486, 'cpu')
        bce = core.TailState(labels, np.zeros(6, dtype=int), np.zeros(6), p, 486, 'cpu')
        model = torch.nn.Linear(1, 1)
        for state in (full, bce):
            state.begin_stage(model)
            state.s[state.positive] += .2
        with torch.no_grad():
            for v in model.parameters():
                v.add_(.3)
        probability = torch.tensor([.7, .4, .1, .9, .3, .5], dtype=torch.float64, requires_grad=True)
        objective_full, info_full = full.block_objective(probability, full.sample(0), 'full')
        objective_bce, info_bce = bce.block_objective(probability, bce.sample(0), 'bce')
        torch.testing.assert_close(objective_full, objective_bce, atol=0, rtol=0)
        self.assertEqual(info_full, info_bce)
        torch.testing.assert_close(full.s, bce.s, atol=0, rtol=0)
        self.assertEqual(float(full.parameter_penalty(model, 'full').detach()), 0.)
        self.assertEqual(float(bce.parameter_penalty(model, 'bce').detach()), 0.)

    def test_training_input_boundary_and_identity_are_preserved(self):
        a, _ = fixture()
        a['anchor_oof'] = np.zeros_like(a['y_train'], dtype=np.float64)
        p = protocol('staco_ranking_v2', core)
        with self.assertRaisesRegex(ValueError, 'training inputs only'):
            revised.fit(dict(a, y_val=np.ones(2)), p, 'full', .001, 1, 'cpu', fixed_epochs=2)
        with self.assertRaisesRegex(ValueError, 'cuda:1'):
            revised.fit(a, p, 'full', .001, 1, 'cuda:0', fixed_epochs=2)
        model, _, trace, detail = revised.fit(dict(a, pair_offset=np.array([0, 14])),
            p, 'full', .001, 1, 'cpu', fixed_epochs=0)
        self.assertEqual(model.state_dict(), {})
        self.assertEqual(trace, [])
        self.assertFalse(detail['trained_neural_model'])


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
