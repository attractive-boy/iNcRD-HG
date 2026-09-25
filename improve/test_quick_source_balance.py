import json
import unittest

import numpy as np
import pandas as pd

from improve import quick_source_balance_modules as balance


class SourceBalanceTests(unittest.TestCase):
    def test_each_disjoint_source_receives_equal_training_mass(self):
        sources = [['paper_a']] * 20 + [['paper_b']] * 2 + [['paper_c']]
        w, loads = balance.source_weights(sources)
        np.testing.assert_allclose([w[:20].sum(), w[20:22].sum(), w[22:].sum()], [23/3]*3)
        self.assertEqual(loads, {'paper_a': 20., 'paper_b': 2., 'paper_c': 1.})

    def test_multisource_triples_receive_fractional_credit(self):
        w, loads = balance.source_weights([['a', 'b'], ['a'], ['b'], ['c']])
        self.assertEqual(loads, {'a': 1.5, 'b': 1.5, 'c': 1.})
        np.testing.assert_allclose(w, np.array([2/3, 2/3, 2/3, 1]) * 4/3)
        reverse, _ = balance.source_weights([['c'], ['b'], ['a'], ['b', 'a']])
        np.testing.assert_array_equal(w, reverse[::-1])

    def test_preserve_class_and_controls_have_intended_mass(self):
        groups = [['a']]*6 + [['b'], ['c']]
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        weights, audit = balance.make_weights(groups, y)
        for c in (0, 1):
            self.assertAlmostEqual(weights['source_balanced_preserve_class'][y == c].sum(), (y == c).sum())
            for name in ('class_prior_only', 'source_weights_shuffled'):
                self.assertAlmostEqual(weights[name][y == c].sum(), weights['source_balanced'][y == c].sum())
            np.testing.assert_array_equal(np.sort(weights['source_weights_shuffled'][y == c]),
                                          np.sort(weights['source_balanced'][y == c]))
        self.assertNotEqual(audit['schemes']['source_balanced']['weighted_resistant_fraction'], y.mean())

    def test_rejects_missing_or_duplicate_provenance(self):
        for groups in ([], [[]], [['a', 'a']], ['null'], ['{}']):
            with self.assertRaises(ValueError):
                balance.source_weights(groups)

    def test_partition_and_triple_leaks_fail(self):
        def frame(source, record, pmid, triple):
            return pd.DataFrame(dict(source_groups=[json.dumps([source])], record_uids=[json.dumps([record])],
                pmids=[json.dumps([pmid])], triple_id=[triple], novel_triple=[True]))
        train = frame('a', 'row1', '1', 't1')
        valid = frame('b', 'row2', '2', 't2')
        self.assertTrue(all(v == 0 for v in balance.audit_partitions(train, valid).values()))
        for candidate in (frame('a', 'row2', '2', 't2'), frame('b', 'row1', '2', 't2'),
                          frame('b', 'row2', '1', 't2'), frame('b', 'row2', '2', 't1')):
            with self.assertRaises(ValueError):
                balance.audit_partitions(train, candidate)


if __name__ == '__main__':
    unittest.main()
