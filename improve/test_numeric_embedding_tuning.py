import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from improve import numeric_embedding as original
from improve import numeric_embedding_tuning as tuning


def row(cfg, seed, score, epoch):
    return dict(config=cfg, config_id=cfg['id'], seed=seed, selection_score=score,
                selected_epoch=epoch, auc=.9, ap=.8, macro_drug_ap=.7)


class TuningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_matched_budget_and_old_anchors_present(self):
        for arm in tuning.ARMS:
            grid = tuning.configurations(arm)
            self.assertEqual(len(grid), 12)
            self.assertEqual(len({c['id'] for c in grid}), 12)
            self.assertEqual({c['lr'] for c in grid}, {.0005, .001, .003})
        base = tuning.configurations('tabm')[4]
        embedded = tuning.configurations('ple_quantile')[7]
        self.assertEqual((base['width'], base['depth'], base['lr'], base['dropout']), (128, 3, .001, .1))
        self.assertEqual((embedded['n_bins'], embedded['embedding_dim'], embedded['lr']), (16, 16, .001))

    def test_confirmation_can_reverse_screen_winner(self):
        a, b = dict(id='a'), dict(id='b')
        rows = [row(a, 666, .99, 90), row(b, 666, .98, 70), row(a, 0, .80, 10),
                row(a, 1, .80, 20), row(b, 0, .97, 40), row(b, 1, .97, 60)]
        self.assertEqual(tuning.rank_screen(rows[:2], 2), [a, b])
        chosen, _ = tuning.rank_confirm([a, b], rows, [666, 0, 1])
        self.assertEqual(chosen['config'], b)
        self.assertEqual(chosen['epochs'], 60)

    def test_duplicates_missing_seeds_and_stable_ties(self):
        a, b = dict(id='a'), dict(id='b')
        self.assertEqual(tuning.rank_screen([row(b, 666, .9, 9), row(a, 666, .9, 10)], 1), [a])
        with self.assertRaises(ValueError):
            tuning.rank_confirm([a], [row(a, 666, .9, 10), row(a, 0, .9, 10)], [666, 0, 1])
        with self.assertRaises(ValueError):
            tuning.rank_confirm([a], [row(a, 666, .9, 10), row(a, 0, .9, 10), row(a, 0, .9, 10)], [666, 0, 1])

    def test_outer_features_require_own_refit_before_any_reference_read(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(tuning.np, 'load') as load:
                with self.assertRaises(FileNotFoundError):
                    tuning.features(Path(directory), {}, 'LncDrug', 'outer')
                load.assert_not_called()
        with self.assertRaises(ValueError):
            tuning.selection('unused', {}, 'outer')

    def test_outer_jobs_use_confirmed_recipe_and_epoch(self):
        chosen = dict(id='b')
        p = dict(tasks=['LncDrug'], arms=['tabm'], refit_seeds=[666, 0, 1])
        def decision(out, name, protocol):
            return dict(chosen={'LncDrug': {'tabm': dict(config=chosen, epochs=27)}})
        with patch.object(tuning.parent, 'decision', side_effect=decision):
            jobs = list(tuning.jobs('unused', p, 'outer'))
        self.assertEqual([j[2] for j in jobs], [chosen]*3)
        self.assertEqual([j[4] for j in jobs], [27]*3)

    def test_worker_assignment_is_disjoint_complete_and_deterministic(self):
        planned = [(task, arm, c, 666, 30) for task in ('LncDrug', 'MiDrug')
                   for arm in tuning.ARMS for c in tuning.configurations(arm)]
        queues = tuning.assign_jobs(planned, 2)
        key = lambda j: (j[0], j[1], j[2]['id'], j[3], j[4])
        actual = [key(j) for q in queues for j in q]
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(set(actual), {key(j) for j in planned})
        self.assertEqual(queues, tuning.assign_jobs(planned, 2))
        self.assertTrue(all(any(j[0] == 'MiDrug' for j in q) for q in queues))

    def test_fast_selection_metrics_preserve_optimizer_and_checkpoint(self):
        rng = np.random.RandomState(12)
        a = dict(x_train=rng.normal(size=(48, 4)).astype(np.float32),
                 x_val=rng.normal(size=(24, 4)).astype(np.float32),
                 y_train=np.array([0., 1., 0.]*16, dtype=np.float32),
                 y_val=np.array([0., 1., 0.]*8, dtype=np.float32),
                 pairs_train=np.vstack([np.arange(48), np.arange(48) % 4]),
                 pairs_val=np.vstack([np.arange(24), np.arange(24) % 4]),
                 feature_manifest_sha256='fixture')
        cfg = dict(id='a', width=12, depth=2, members=3, dropout=.1, lr=.001,
                   weight_decay=1e-4, n_bins=4, embedding_dim=4, min_leaf=4)
        p = dict(batch_size=16, smoothing_window=2, sha256='fixture')
        for arm in ('tabm', 'ple_quantile'):
            with contextlib.redirect_stdout(io.StringIO()):
                old = original.train(a, 'LncDrug', arm, cfg, 666, 'screen', 6, p, 'cpu')
                new = tuning.train(a, 'LncDrug', arm, cfg, 666, 'screen', 6, p, 'cpu')
            self.assertEqual(old[0]['selected_epoch'], new[0]['selected_epoch'])
            np.testing.assert_array_equal(old[1], new[1])
            for x, y in zip(old[2], new[2]):
                for metric in ('loss', 'auc', 'ap', 'criterion'):
                    self.assertEqual(x[metric], y[metric])
            for key in old[3]:
                torch.testing.assert_close(old[3][key], new[3][key], atol=0, rtol=0)

    def test_gate_rejects_single_metric_or_single_seed_gain(self):
        self.assertFalse(tuning.gate([dict(auc=.001, ap=-.001)]*3))
        self.assertFalse(tuning.gate([dict(auc=.01, ap=.01), dict(auc=-.001, ap=-.001), dict(auc=-.001, ap=-.001)]))
        self.assertTrue(tuning.gate([dict(auc=.001, ap=.001)]*3))


if __name__ == '__main__':
    unittest.main()
