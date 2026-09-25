"""Reject context contamination while respecting TabICL's constant filter."""
from types import SimpleNamespace
import unittest

import numpy as np
from tabicl._sklearn.preprocessing import UniqueFeatureFilter

from improve.nested_anchor_fit_v3 import audit_training_context


def model_for(x, y):
    feature_filter = UniqueFeatureFilter().fit(x)
    return SimpleNamespace(X_encoder_=SimpleNamespace(transform=lambda value: value.copy()),
        ensemble_generator_=SimpleNamespace(unique_filter_=feature_filter,
            X_=feature_filter.transform(x), y_=y.copy()))


class ConstantFeatureAuditTests(unittest.TestCase):
    def test_correct_training_only_constant_removal_and_reload(self):
        x = np.array([[1., 7., 3.], [2., 7., 4.], [3., 7., 5.]])
        y = np.array([0, 1, 0])
        model = model_for(x, y)
        record = audit_training_context(model, x, y)
        self.assertEqual(record['removed_columns'], [1])
        self.assertEqual(record['kept_columns'], 2)
        with self.assertRaises(AssertionError):
            audit_training_context(model, x[::-1], y)
        with self.assertRaises(AssertionError):
            audit_training_context(model, x, 1 - y)

    def test_query_conditioned_or_wrong_mask_is_rejected(self):
        x = np.array([[1., 7., 3.], [2., 7., 4.], [3., 7., 5.]])
        y = np.array([0, 1, 0])
        model = model_for(x, y)
        # A query value may vary in column 1; fitting the filter on it is invalid.
        poisoned = np.concatenate([x, [[9., 100., 0.]]])
        model.ensemble_generator_.unique_filter_ = UniqueFeatureFilter().fit(poisoned)
        model.ensemble_generator_.X_ = x.copy()
        with self.assertRaises(AssertionError):
            audit_training_context(model, x, y)

    def test_constant_and_single_row_fallbacks_match_estimator(self):
        for x in (np.ones((3, 4)), np.array([[1., 2., 3., 4.]])):
            y = np.zeros(len(x), dtype=int)
            record = audit_training_context(model_for(x, y), x, y)
            self.assertEqual(record['kept_columns'], 1 if len(x) > 1 else 4)


if __name__ == '__main__':
    unittest.main()
