"""Guard primary choice, paired baselines, missing fits and joint metric gates."""
import copy
import unittest

from improve.assess_declared_graph import evaluate


class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.p = dict(method='residual_geometry', tasks=['MiDrug', 'LncDrug'],
                      kinds=['full', 'euclidean', 'attention_only'],
                      primary_kind='attention_only', primary_controls=['euclidean'])
        self.seeds = [666, 0, 1]
        self.records = {}
        for task in self.p['tasks']:
            for kind, delta in [('full', .003), ('euclidean', .001), ('attention_only', .002)]:
                for seed in self.seeds:
                    baseline = dict(auc=.9, ap=.8)
                    value = delta if task == 'MiDrug' else -.0001
                    self.records[task, kind, seed] = dict(metrics=dict(
                        anchor=baseline, anchor_composed={m: v + value for m, v in baseline.items()}),
                        offset_metrics=dict(offset_composed={m: v + .0005 for m, v in baseline.items()}))

    def test_declared_primary_is_not_the_full_label(self):
        r = evaluate(self.p, self.records, self.seeds)
        self.assertTrue(r['declared_gate'])
        self.assertLess(r['primary_minus_controls']['MiDrug']['full']['auc'], 0)

    def test_one_lnc_loss_above_tolerance_fails(self):
        self.records['LncDrug', 'attention_only', 0]['metrics']['anchor_composed']['ap'] = .7989
        self.assertFalse(evaluate(self.p, self.records, self.seeds)['performance_gate'])

    def test_mechanism_requires_both_metrics(self):
        for seed in self.seeds:
            self.records['MiDrug', 'attention_only', seed]['metrics']['anchor_composed']['auc'] = .9009
        r = evaluate(self.p, self.records, self.seeds)
        self.assertTrue(r['performance_gate'])
        self.assertFalse(r['primary_mechanism_gate'])
        self.assertFalse(r['declared_gate'])

    def test_incomplete_or_mismatched_records_are_rejected(self):
        missing = copy.deepcopy(self.records)
        del missing['MiDrug', 'full', 0]
        with self.assertRaisesRegex(ValueError, 'Missing'):
            evaluate(self.p, missing, self.seeds)
        self.records['MiDrug', 'full', 0]['metrics']['anchor']['auc'] = .91
        with self.assertRaisesRegex(ValueError, 'Unpaired'):
            evaluate(self.p, self.records, self.seeds)

    def test_offset_null_is_mandatory_and_must_be_beaten(self):
        for r in self.records.values():
            r['offset_metrics']['offset_composed'] = dict(auc=.904, ap=.804)
        self.assertFalse(evaluate(self.p, self.records, self.seeds)['offset_null_gate'])
        for r in self.records.values():
            del r['offset_metrics']
        with self.assertRaisesRegex(ValueError, 'require an offset'):
            evaluate(self.p, self.records, self.seeds)


if __name__ == '__main__':
    unittest.main()
