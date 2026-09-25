import contextlib
import io
import unittest

import numpy as np
import torch

from improve import global_ap
from improve import numeric_embedding as original
from improve.global_ap_modules import positive_anchors, smooth_ap_loss


class GlobalAPTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    @staticmethod
    def fixture():
        rng = np.random.RandomState(12)
        a = dict(x_train=rng.normal(size=(48, 4)).astype(np.float32),
                 x_val=rng.normal(size=(24, 4)).astype(np.float32),
                 y_train=np.array([0., 1., 0.]*16, dtype=np.float32),
                 y_val=np.array([0., 1., 0.]*8, dtype=np.float32),
                 pairs_train=np.vstack([np.arange(48), np.arange(48) % 4]),
                 pairs_val=np.vstack([np.arange(24), np.arange(24) % 4]),
                 feature_manifest_sha256='fixture')
        cfg = dict(id='a', width=12, depth=2, members=3, dropout=.1, lr=.001,
                   weight_decay=1e-4, ap_lambda=0., temperature=.05,
                   n_bins=4, embedding_dim=4, min_leaf=4)
        p = dict(batch_size=16, smoothing_window=2, sha256='fixture',
                 ap_anchor_budget=12, ap_warmup_epochs=2)
        return a, cfg, p

    def test_zero_rank_weight_reproduces_old_training_exactly(self):
        a, cfg, p = self.fixture()
        with contextlib.redirect_stdout(io.StringIO()):
            before = original.train(a, 'LncDrug', 'tabm', cfg, 666, 'screen', 6, p, 'cpu')
            after = global_ap.train(a, 'LncDrug', 'smooth_ap', cfg, 666, 'screen', 6, p, 'cpu')
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[0]['selected_epoch'], after[0]['selected_epoch'])
        for key in before[3]:
            torch.testing.assert_close(before[3][key], after[3][key], atol=0, rtol=0)

    def test_ap_changes_updates_without_changing_dropout_rng_consumption(self):
        a, cfg, p = self.fixture()
        with contextlib.redirect_stdout(io.StringIO()):
            before = global_ap.train(a, 'LncDrug', 'tabm', cfg, 666, 'screen', 6, p, 'cpu')
            rng_before = torch.random.get_rng_state().clone()
            after = global_ap.train(a, 'LncDrug', 'smooth_ap', dict(cfg, ap_lambda=.5),
                                    666, 'screen', 6, p, 'cpu')
            rng_after = torch.random.get_rng_state().clone()
        torch.testing.assert_close(rng_before, rng_after, atol=0, rtol=0)
        self.assertGreater(float(np.max(np.abs(before[1]-after[1]))), 1e-7)
        self.assertTrue(all(np.isfinite(r['ap_surrogate']) for r in after[2]))
        self.assertEqual(after[2][0]['ap_weight'], .25)
        self.assertEqual(after[2][1]['ap_weight'], .5)
        self.assertEqual(after[2][-1]['ap_weight'], .5)

    def test_consensus_ranking_propagates_to_every_member(self):
        torch.manual_seed(42)
        logits = torch.randn(18, 5, requires_grad=True)
        y = torch.tensor([0., 1., 0.]*6)
        probabilities = torch.sigmoid(logits).mean(1)
        loss = smooth_ap_loss(probabilities, y, positive_anchors(y, 6, 3))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad.abs().sum(0) > 0).all())

    def test_loss_search_budget_and_temperature_grid(self):
        for arm in global_ap.ARMS:
            configs = global_ap.configurations(arm)
            self.assertEqual(len(configs), 12)
            self.assertEqual(len({c['id'] for c in configs}), 12)
        cfgs = global_ap.configurations('smooth_ap')
        self.assertEqual({c['temperature'] for c in cfgs}, {.01, .05})
        self.assertEqual({c['ap_lambda'] for c in cfgs}, {.1, .5, 1.})
        self.assertTrue(all(c['width'] == 128 and c['depth'] == 3 for c in cfgs))


if __name__ == '__main__':
    unittest.main()
