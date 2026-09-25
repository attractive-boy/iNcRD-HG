"""Selection integrity checks for the expanded tuning experiment."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from improve import structure_program_tuning as tuning


def row(cfg, seed, score, epoch, auc=.8, ap=.7):
    return dict(config=cfg, config_id=cfg['id'], seed=seed, selection_score=score,
        selected_epoch=epoch, auc=auc, ap=ap, macro_drug_ap=.6)


class TuningSelectionTests(unittest.TestCase):
    def test_confirmation_can_reverse_screen_winner(self):
        a, b = dict(id='a'), dict(id='b')
        rows = [row(a, 666, .99, 90), row(b, 666, .98, 70),
                row(a, 0, .80, 10), row(a, 1, .80, 20),
                row(b, 0, .97, 40), row(b, 1, .97, 60)]
        self.assertEqual(tuning.rank_screen(rows[:2], 2), [a, b])
        chosen, summaries = tuning.rank_confirm([a, b], rows, [666, 0, 1])
        self.assertEqual(chosen['config'], b)
        self.assertEqual(chosen['epochs'], 60)
        self.assertEqual(len(summaries), 2)

    def test_selection_uses_frozen_objective_and_stable_tie(self):
        a, b = dict(id='a'), dict(id='b')
        rows = [row(b, 666, .91, 8, .99, .99), row(a, 666, .92, 9, .8, .8)]
        self.assertEqual(tuning.rank_screen(rows, 1), [a])
        rows[0]['selection_score'] = .92
        self.assertEqual(tuning.rank_screen(rows, 1), [a])
        chosen, _ = tuning.rank_confirm([b, a], rows, [666])
        self.assertEqual(chosen['config'], a)

    def test_missing_or_duplicate_seeds_rejected(self):
        cfg = dict(id='a')
        for seeds in ([666, 0], [666, 0, 0]):
            with self.assertRaises(ValueError):
                tuning.rank_confirm([cfg], [row(cfg, s, .9, 10) for s in seeds], [666, 0, 1])

    def test_outer_features_require_new_decision_before_reading_reference(self):
        with tempfile.TemporaryDirectory() as path:
            with patch.object(tuning.original, 'protocol') as old_protocol:
                with self.assertRaises(FileNotFoundError):
                    tuning.features(Path(path), dict(arms=['base']), 'LncDrug', 'outer', 'base')
                old_protocol.assert_not_called()

    def test_outer_jobs_use_confirmed_config_and_median_epochs(self):
        a, b = dict(id='screen_winner'), dict(id='confirmed_winner')
        p = dict(tasks=['LncDrug'], arms=['base'], refit_seeds=[666, 0, 1])
        def decision(out, name, protocol):
            if name == 'PARAMETERS.json':
                return dict(chosen={'LncDrug': {'base': [a, b]}})
            return dict(chosen={'LncDrug': {'base': dict(config=b, epochs=23)}})
        with patch.object(tuning.parent, 'decision', side_effect=decision):
            jobs = list(tuning.jobs('unused', p, 'outer'))
        self.assertEqual([j[2] for j in jobs], [b, b, b])
        self.assertEqual([j[4] for j in jobs], [23, 23, 23])

    def test_outer_cannot_select(self):
        with self.assertRaises(ValueError):
            tuning.selection('unused', {}, 'outer')

    def test_seed_alignment_and_single_metric_gain(self):
        def frame(seeds):
            return pd.DataFrame([dict(task='T', arm='A', seed=s, auc=.9,
                ap=.8, macro_drug_ap=.7, macro_recall10_sampled=.6) for s in seeds])
        with self.assertRaises(ValueError):
            tuning.compare(frame([666, 0, 1]), frame([666, 0, 2]), 'T', 'A', 'A', 'check')
        self.assertFalse(tuning.performance_gate([dict(auc=-.001, ap=.01, macro_drug_ap=.01)]*3))
        self.assertFalse(tuning.performance_gate([dict(auc=.001, ap=.01, macro_drug_ap=-.01)]*3))
        self.assertTrue(tuning.performance_gate([dict(auc=.001, ap=.01, macro_drug_ap=.01)]*3))


if __name__ == '__main__':
    unittest.main()
