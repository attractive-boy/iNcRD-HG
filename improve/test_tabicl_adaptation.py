"""Checks for training-label isolation, frozen encoders, and selection boundaries."""
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from improve import tabicl_adaptation as experiment
from improve import tabicl_adaptation_modules as kernels


class TinyContextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.col_embedder = nn.Linear(3, 4)
        self.row_interactor = nn.Linear(4, 4)
        self.icl_predictor = nn.Linear(4, 2)
        self.forward_labels = None
        self.forward_logits = None

    def forward(self, x, context_y):
        self.forward_labels = context_y.detach().clone()
        x = x[:, context_y.shape[1]:]
        logits = self.icl_predictor(torch.tanh(self.row_interactor(torch.tanh(self.col_embedder(x)))))
        self.forward_logits = logits.detach().clone()
        return logits


class AdaptationChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.model = TinyContextModel()
        self.batch = kernels.official_meta_module().MetaBatch(
            X=torch.randn(2, 30, 3), y_train=torch.tensor([[0., 1.]*10]*2),
            y_query=torch.tensor([[0, 1]*5]*2), train_size=20)

    def optimizer(self, model):
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)

    def test_query_labels_are_only_used_in_loss(self):
        other = TinyContextModel()
        other.load_state_dict(self.model.state_dict())
        changed = kernels.official_meta_module().MetaBatch(X=self.batch.X, y_train=self.batch.y_train,
            y_query=1-self.batch.y_query, train_size=self.batch.train_size)
        kernels.update(self.model, self.batch, self.optimizer(self.model), 'cpu')
        kernels.update(other, changed, self.optimizer(other), 'cpu')
        torch.testing.assert_close(self.model.forward_labels, self.batch.y_train)
        self.assertEqual(self.model.forward_labels.shape[1], 20)
        torch.testing.assert_close(self.model.forward_logits, other.forward_logits, rtol=0, atol=0)
        self.assertFalse(torch.equal(self.model.icl_predictor.weight, other.icl_predictor.weight))

    def test_icl_adaptation_preserves_both_encoders(self):
        before = kernels.group_digests(self.model)
        count = kernels.set_trainable(self.model, 'icl')
        self.assertEqual(count, sum(p.numel() for p in self.model.icl_predictor.parameters()))
        kernels.update(self.model, self.batch, self.optimizer(self.model), 'cpu')
        after = kernels.group_digests(self.model)
        for key in ('col_embedder', 'row_interactor'):
            self.assertEqual(before[key], after[key])
        self.assertNotEqual(before['icl_predictor'], after['icl_predictor'])
        self.assertTrue(self.model.col_embedder.training)

    def test_all_adaptation_updates_encoders(self):
        before = kernels.group_digests(self.model)
        kernels.set_trainable(self.model, 'all')
        kernels.update(self.model, self.batch, self.optimizer(self.model), 'cpu')
        after = kernels.group_digests(self.model)
        for key in before:
            self.assertNotEqual(before[key], after[key])

    def test_invalid_context_query_boundary_fails(self):
        self.batch.train_size = 19
        with self.assertRaisesRegex(ValueError, 'boundary'):
            kernels.update(self.model, self.batch, self.optimizer(self.model), 'cpu')

    def test_nonfinite_update_is_rejected(self):
        self.batch.X[0, -1, 0] = np.nan
        with self.assertRaisesRegex(FloatingPointError, 'loss'):
            kernels.update(self.model, self.batch, self.optimizer(self.model), 'cpu')

    def test_validation_preserves_all_random_generators(self):
        np.random.seed(3)
        random.seed(4)
        torch.manual_seed(5)
        initial = (np.random.get_state(), random.getstate(), torch.get_rng_state())
        with kernels.inference_rng_guard('cpu'):
            np.random.random(10)
            random.random()
            torch.rand(10)
        after = (np.random.random(), random.random(), torch.rand(3))
        np.random.set_state(initial[0])
        random.setstate(initial[1])
        torch.set_rng_state(initial[2])
        expected = (np.random.random(), random.random(), torch.rand(3))
        self.assertEqual(after[:2], expected[:2])
        torch.testing.assert_close(after[2], expected[2], rtol=0, atol=0)

    def test_outer_features_require_own_frozen_selection(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(experiment.base, 'load_features') as loader:
            with self.assertRaises(FileNotFoundError):
                experiment.features(Path(folder), dict(tasks=['LncDrug']), 'LncDrug', 'outer')
            loader.assert_not_called()

    def test_all_seeds_choose_mode_and_step(self):
        candidates = [dict(id='frozen', mode='frozen', steps=0), dict(id='icl_008', mode='icl', steps=8)]
        rows = [dict(recipe_id=cfg['id'], seed=seed, auc=value, ap=value, criterion=value, macro_drug_ap=value)
                for cfg, values in zip(candidates, ((.8, .8, .8), (.99, .6, .6)))
                for seed, value in zip((666, 0, 1), values)]
        self.assertEqual(experiment.choose(rows, candidates, [666, 0, 1])['recipe'], candidates[0])
        rows[-1]['seed'] = 0
        with self.assertRaisesRegex(ValueError, 'Missing or duplicate'):
            experiment.choose(rows, candidates, [666, 0, 1])

    def test_exact_ties_keep_unadapted_model(self):
        candidates = [dict(id='all_008', mode='all', steps=8), dict(id='frozen', mode='frozen', steps=0)]
        rows = [dict(recipe_id=cfg['id'], seed=s, auc=.8, ap=.8, criterion=.8, macro_drug_ap=None)
                for cfg in candidates for s in (666, 0, 1)]
        self.assertEqual(experiment.choose(rows, candidates, [666, 0, 1])['recipe']['steps'], 0)

    def test_outer_refit_only_uses_fitting_rows_and_fixed_horizon(self):
        a = dict(x_train=np.ones((60, 3)), y_train=np.array([0, 1]*30), x_val=object())
        p = dict(input_configs={'LncDrug': {'input_view': 'quantile'}}, learning_rate=1e-5,
                 weight_decay=.01, horizon_steps=48, chunk_size=2048, training_members=2)
        clf = SimpleNamespace(model_=self.model)
        rows = iter([(object(), 0)]*3)
        snapshots = []
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(experiment, 'new_estimator', return_value=clf), \
             patch.object(kernels, 'meta_batches', return_value=rows) as build, \
             patch.object(kernels, 'update', return_value=dict(loss=.5, grad_norm=.2, context_rows=40, query_rows=20)) as update, \
             patch.object(experiment, 'snapshot', side_effect=lambda *args: snapshots.append(list(args[8]))), \
             patch.object(torch.cuda, 'reset_peak_memory_stats'), patch.object(torch.cuda, 'empty_cache'):
            experiment.train_trajectory(Path(folder), p, 'LncDrug', 'outer', 'icl', 666, [3], a, 'cpu')
        self.assertIs(build.call_args.args[0], a['x_train'])
        self.assertIs(build.call_args.args[1], a['y_train'])
        self.assertEqual(update.call_count, 3)
        self.assertEqual(len(snapshots), 1)
        self.assertEqual([r['learning_rate'] for r in snapshots[0]], [kernels.learning_rate(i, 48, 1e-5) for i in range(3)])

    def test_selection_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            p = dict(sha256='protocol', tasks=['LncDrug'], modes=['icl', 'all'], milestones=[8, 24, 48])
            (out/'inner_results.csv').write_text('verified inner results\n')
            value = experiment.base.seal(dict(protocol_sha256=p['sha256'], chosen={'LncDrug': {'recipe':experiment.recipes(p)[0]}},
                artifacts_sha256={'inner_results.csv':experiment.base.digest(out/'inner_results.csv')}))
            experiment.base.write_json(out/'SELECTION.json', value)
            experiment.selection(out, p)
            (out/'inner_results.csv').write_text('altered results\n')
            with self.assertRaisesRegex(ValueError, 'selection input'):
                experiment.selection(out, p)


if __name__ == '__main__':
    unittest.main()
