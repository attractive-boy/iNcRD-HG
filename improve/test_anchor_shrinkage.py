"""Unit checks for the anchor-shrinkage recipes (no heavy fitting)."""
import unittest

import numpy as np
from scipy.special import expit

from improve import anchor_shrinkage as anchor


class AnchorShrinkageTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 200
        self.ref = rng.uniform(0.05, 0.95, n)
        self.anchor = dict(
            pool=rng.normal(size=n), pool_shuffled=rng.normal(size=n),
            et=rng.normal(size=n), pu=rng.normal(size=n))

    def test_logit_recipe_matches_formula(self):
        value, _ = anchor.recipes(self.anchor, self.ref)['anchor_logit']
        expected = expit((1 - anchor.POOL_WEIGHT) * anchor.logit(self.ref) + anchor.POOL_WEIGHT * self.anchor['pool'])
        np.testing.assert_allclose(value, expected, rtol=0, atol=1e-12)

    def test_rank_recipe_is_rank_mean(self):
        value, _ = anchor.recipes(self.anchor, self.ref)['anchor_rank']
        self.assertGreaterEqual(value.min(), 0.0)
        self.assertLessEqual(value.max(), 1.0)
        expected_rank = (1 - anchor.POOL_WEIGHT) * anchor.zrank(self.ref) + anchor.POOL_WEIGHT * anchor.zrank(self.anchor['pool'])
        np.testing.assert_allclose(value, expected_rank, rtol=0, atol=1e-12)

    def test_single_weight_recipes(self):
        recipes = anchor.recipes(self.anchor, self.ref)
        for name, key in (('anchor_pu', 'pu'), ('anchor_et', 'et')):
            value, _ = recipes[name]
            expected = expit((1 - anchor.SINGLE_WEIGHT) * anchor.logit(self.ref) + anchor.SINGLE_WEIGHT * self.anchor[key])
            np.testing.assert_allclose(value, expected, rtol=0, atol=1e-12)

    def test_controls_use_shuffled_pool(self):
        recipes = anchor.recipes(self.anchor, self.ref)
        logit_control = expit((1 - anchor.POOL_WEIGHT) * anchor.logit(self.ref) + anchor.POOL_WEIGHT * self.anchor['pool_shuffled'])
        np.testing.assert_allclose(recipes['anchor_logit_shuffled'][0], logit_control, rtol=0, atol=1e-12)
        self.assertFalse(np.allclose(recipes['anchor_logit_shuffled'][0], recipes['anchor_logit'][0]))

    def test_logit_inverse(self):
        np.testing.assert_allclose(expit(anchor.logit(self.ref)), self.ref, rtol=0, atol=1e-9)

    def test_mixed_recipe_is_pool_only_for_lncdrug(self):
        tdp = np.zeros_like(self.ref)
        mixed = anchor.mixed_recipe(self.anchor, self.ref, 'LncDrug', tdp)
        expected = expit((1 - anchor.POOL_WEIGHT) * anchor.logit(self.ref) + anchor.POOL_WEIGHT * self.anchor['pool'])
        np.testing.assert_allclose(mixed, expected, rtol=0, atol=1e-12)

    def test_mixed_recipe_adds_tabdpt_for_midrug(self):
        tdp = np.zeros_like(self.ref)
        mixed = anchor.mixed_recipe(self.anchor, self.ref, 'MiDrug', tdp)
        weight = anchor.POOL_WEIGHT + anchor.MIXED_TDP_WEIGHT
        expected = expit((1 - weight) * anchor.logit(self.ref) + anchor.POOL_WEIGHT * self.anchor['pool'])
        np.testing.assert_allclose(mixed, expected, rtol=0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
