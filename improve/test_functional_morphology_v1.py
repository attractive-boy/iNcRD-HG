"""Verify exact promotion and the narrower control before any new scores."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from improve import anchor_morphological_alignment_v1 as old
from improve import anchor_functional_morphology_v1 as new
from improve import morphological_alignment_v1 as parent
from improve import functional_morphology_v1 as core
from improve import test_morphological_alignment_v1 as fixtures
from improve import tabular_foundation as base


def same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return set(left) == set(right) and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


class RevisionChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        fixtures.fixture(self.folder)
        self.c1, self.c2 = parent.parameters(self.folder), core.parameters(self.folder)
        for c in (self.c1, self.c2):
            c.update(width=8, hidden=12, dti_batch=16)
        self.p1 = dict(smoke=True, implementation_module='morphological_alignment_v1',
            source_sha256={'morphological_alignment_v1.py': base.digest(parent.__file__)},
            core_parameters=self.c1, kinds=list(parent.KINDS), epochs=3, check_interval=20,
            weight_decay=.001, strong_weight=.1, private_rng_offset=32017)
        self.p2 = dict(self.p1, implementation_module='functional_morphology_v1', core_parameters=self.c2,
            source_sha256={'functional_morphology_v1.py': base.digest(core.__file__)}, kinds=list(core.KINDS))
        torch.set_num_threads(2)

    def tearDown(self):
        self.temp.cleanup()

    def test_parent_and_revised_full_match_all_training_state(self):
        a = fixtures.training()
        m1, _, trace1, _ = old.fit(a, self.p1, 'no_alignment', .001, 32100, 'cpu', fixed_epochs=3)
        rng1 = torch.random.get_rng_state().clone()
        m2, _, trace2, _ = new.fit(a, self.p2, 'full', .001, 32100, 'cpu', fixed_epochs=3)
        self.assertTrue(torch.equal(rng1, torch.random.get_rng_state()))
        self.assertTrue(same(m1.state_dict(), m2.state_dict()))
        self.assertTrue(same(m1.training_final_state, m2.training_final_state))
        self.assertEqual(trace1, trace2)
        self.assertEqual(m2.kind, 'no_alignment')

    def test_new_control_is_zero_alignment_with_matched_budget(self):
        a = fixtures.training()
        m1, _, tr1, r1 = new.fit(a, self.p2, 'full', .001, 32101, 'cpu', fixed_epochs=3)
        rng1 = torch.random.get_rng_state().clone()
        m2, _, tr2, r2 = new.fit(a, self.p2, 'shuffled_rna_targets', .001, 32101, 'cpu', fixed_epochs=3)
        self.assertEqual(m2.kind, 'shuffled_rna_targets')
        self.assertTrue(all(row['alignment_multiplier'] == 0. for row in tr1 + tr2))
        self.assertEqual(r1['parameters'], r2['parameters'])
        self.assertEqual(r2['training_steps'], 3)
        self.assertTrue(torch.equal(rng1, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(m1.training_final_state['private_rng'], m2.training_final_state['private_rng']))

    def test_parent_hash_and_zero_auxiliary_weight_are_enforced(self):
        a = fixtures.training()
        with self.assertRaises(ValueError):
            core.setup(a, dict(self.c2, alignment_weight=.1), 'full', 'cpu')
        with self.assertRaises(ValueError):
            core.setup(a, dict(self.c2, parent_core_sha256='incorrect'), 'full', 'cpu')
        m, xs, graphs, _ = core.setup(fixtures.training(lnc=True), self.c2, 'full', 'cpu')
        self.assertEqual(list(m.parameters()), [])
        self.assertEqual(xs, {})
        self.assertEqual(graphs, [None] * 5)


if __name__ == '__main__':
    unittest.main()
