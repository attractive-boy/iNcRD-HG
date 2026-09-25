"""Independent source, optimizer, hypergradient and label-boundary checks."""
from collections import OrderedDict
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from improve import anchor_meta_loss_experiment_v1 as experiment
from improve import meta_loss_graph_v1 as core
from improve.test_nested_anchor_v2 import fixture


def training_fixture():
    a, _ = fixture()
    a['anchor_oof'] = np.linspace(-1., 1., len(a['y_train']))
    c = core.parameters()
    c.update(width=8, heads=2, depth=1)
    p = dict(smoke=True, family='meta_loss', implementation_module='meta_loss_graph_v1',
        source_sha256={'meta_loss_graph_v1.py': experiment.common.base.digest(core.__file__)},
        core_parameters=c, kinds=list(core.KINDS), epochs=7, check_interval=5,
        weight_decay=.001, strong_weight=.1, private_rng_offset=29015)
    return a, p


class MetaLossTests(unittest.TestCase):
    def test_features_match_pinned_author_statistics_and_rolling_memory(self):
        path = Path(__file__).parent / 'research/gradient_selection_next_20260916/storm_author_v1_utils_storm.py'
        spec = importlib.util.spec_from_file_location('reviewed_storm_util', path)
        source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source)
        memory, author = core.FeatureMemory(8), source.Filter(2, 1, 8)
        labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
        for step in range(3):
            logits = torch.tensor([[-1.2, .9, 1.4, -.7, -.8, 1.1, .9, -1.3],
                                   [-.9, 1.2, 1.1, -.6, -1., 1.4, 1.1, -1.2],
                                   [-1.3, 1.3, 1., -.9, -.6, .9, 1.2, -1.]], dtype=torch.float64)
            logits = logits * (1 + .15 * step)
            losses = F.binary_cross_entropy_with_logits(logits, labels.double().expand_as(logits), reduction='none')
            stats = source.Results(losses, torch.stack((torch.zeros_like(logits), logits), -1), labels)
            author.update_batch_stats(stats)
            batch = dict(input_ids=torch.zeros(8, 1), labels=labels)
            kls, ovls = author.get_sample_stats(batch, stats)
            expected = []
            for i, label in enumerate(labels):
                l, cat = int(label), 2 - int(stats['agreement'][i])
                def group(key):
                    return author.get_stats(key)[l][cat]
                expected.append([float(stats['losses_means'][i]), group('loss_means_mean'), group('loss_means_std'),
                    float(stats['losses_stds'][i]), group('loss_stds_mean'), group('loss_stds_std'),
                    float(stats['probs_means'][i]), group('prob_means_mean'), group('prob_means_std'),
                    float(stats['probs_stds'][i]), group('prob_stds_mean'), group('prob_stds_std'),
                    kls[i], group('kl_mean'), group('kl_std'), ovls[i], group('ovl_mean'), group('ovl_std'),
                    float(stats['agreement'][i])])
            actual = memory.features(logits.numpy(), losses.numpy(), labels.numpy(), np.arange(8) + 8 * step)
            np.testing.assert_allclose(actual, np.array(expected, dtype=np.float32), atol=2e-5, rtol=2e-5)
        self.assertEqual([len(v) for v in memory.rows.values()], [8, 8])
        restored = core.FeatureMemory(8)
        restored.load_state_dict(memory.state_dict())
        for c in (0, 1):
            np.testing.assert_array_equal(restored.rows[c], memory.rows[c])
        zeros = np.zeros((3, 8))
        edge_features = core.FeatureMemory(8).features(zeros, zeros + np.log(2.), labels.numpy(), np.arange(8))
        self.assertTrue(np.isfinite(edge_features).all())
        self.assertTrue(np.all(edge_features[:, 12] == 0.))
        # Exactly identical dropout observations, with nontrivial FP64 values,
        # must have zero variance rather than a rounding-sized positive value.
        constant = np.array([.0119, .08531, .278417, .456121, 1.003817, 1.438817, 3.1415926, 7.623199])
        repeated = np.tile(constant, (3, 1))
        np.testing.assert_array_equal(core._sample_std(repeated, axis=0), np.zeros(8))
        stable = core.FeatureMemory(8).features(repeated, repeated, labels.numpy(), np.arange(8))
        np.testing.assert_array_equal(stable[:, 3], np.zeros(8))
        np.testing.assert_array_equal(stable[:, 12], np.zeros(8))

    def test_differentiable_adamw_matches_torch_with_persistent_moments(self):
        torch.random.default_generator.manual_seed(429)
        theta = OrderedDict(w=torch.randn(3, 2, dtype=torch.float64, requires_grad=True),
                            b=torch.randn(2, dtype=torch.float64, requires_grad=True))
        expected = [torch.nn.Parameter(t.detach().clone()) for t in theta.values()]
        optimizer = torch.optim.AdamW(expected, lr=.003, weight_decay=.01, foreach=False)
        state = {}
        for step in range(6):
            gs = {k: torch.randn_like(v) for k, v in theta.items()}
            gs['w'][0, 0] = 0.
            if step == 3:
                gs['b'].zero_()
            updated, next_state, _ = core.differentiable_adamw(theta, gs, state,
                lr=.003, weight_decay=.01, clip=.4)
            for target, g in zip(expected, gs.values()):
                target.grad = g.clone()
            torch.nn.utils.clip_grad_norm_(expected, .4)
            optimizer.step()
            for i, (name, value) in enumerate(updated.items()):
                torch.testing.assert_close(value, expected[i], atol=2e-14, rtol=2e-14)
                torch.testing.assert_close(next_state['first'][name], optimizer.state[expected[i]]['exp_avg'], atol=2e-14, rtol=2e-14)
                torch.testing.assert_close(next_state['second'][name], optimizer.state[expected[i]]['exp_avg_sq'], atol=2e-14, rtol=2e-14)
            state = core.detached_optimizer_state(next_state)
            theta = OrderedDict((k, v.detach().requires_grad_()) for k, v in updated.items())
        self.assertEqual(state['step'], 6)
        self.assertTrue(all(not v.requires_grad for v in state['first'].values()))

    def test_two_meta_gradient_paths_match_finite_differences(self):
        x = torch.tensor([[.3, -.7], [1.2, .2], [-.8, -.2], [.4, 1.3]], dtype=torch.float64)
        q = torch.tensor([[.7, .5], [-1.1, .3], [.2, -.6]], dtype=torch.float64)
        y, yq = torch.tensor([0., 1., 0., 1.], dtype=torch.float64), torch.tensor([1., 0., 1.], dtype=torch.float64)
        fi, fm = torch.tensor([[1., .2], [.3, 1.], [-.3, .4], [.9, -.5]], dtype=torch.float64), q.clone()
        def objective(omega, mode='full', return_paths=False):
            theta = torch.tensor([.2, -.4], dtype=torch.float64, requires_grad=True)
            wi = 2 * (fi @ omega).sigmoid()
            risks = F.binary_cross_entropy_with_logits(x @ theta, y, reduction='none')
            g = torch.autograd.grad((wi * risks).mean(), theta, create_graph=True)[0]
            state = dict(step=4, first={'w': torch.tensor([.03, -.04], dtype=torch.float64)},
                         second={'w': torch.tensor([.01, .02], dtype=torch.float64)})
            updated, _, _ = core.differentiable_adamw({'w': theta}, {'w': g}, state,
                lr=.03, weight_decay=.01)
            value = updated['w'].detach() if mode == 'no_meta_gradient' else updated['w']
            meta_risk = F.binary_cross_entropy_with_logits(q @ value, yq, reduction='none')
            wm = 2 * (fm @ omega).sigmoid()
            if mode == 'no_outer_rescaling':
                wm = torch.ones_like(wm) + 0. * wm
            loss = (wm * meta_risk).mean()
            if return_paths:
                direct = torch.autograd.grad((wm * meta_risk.detach()).mean(), omega, retain_graph=True)[0]
                indirect = torch.autograd.grad((wm.detach() * meta_risk).mean(), omega, retain_graph=True)[0]
                total = torch.autograd.grad(loss, omega)[0]
                return total, direct, indirect
            return loss
        omega = torch.tensor([.3, -.5], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(objective, (omega,), atol=1e-7, rtol=1e-5))
        total, direct, indirect = objective(omega, return_paths=True)
        torch.testing.assert_close(total, direct + indirect, atol=1e-12, rtol=1e-12)
        self.assertGreater(float(direct.norm()), .01)
        self.assertGreater(float(indirect.norm()), 1e-5)
        ablated = torch.autograd.grad(objective(omega, 'no_meta_gradient'), omega)[0]
        torch.testing.assert_close(ablated, direct, atol=1e-12, rtol=1e-12)
        # An inactive task coordinate must not produce a NaN hypergradient.
        zero_g = {'w': omega * 0.}
        updated, _, _ = core.differentiable_adamw({'w': omega}, zero_g, {}, lr=.003, weight_decay=.001)
        self.assertTrue(torch.isfinite(torch.autograd.grad(updated['w'].sum(), omega)[0]).all())

    def test_disjoint_episode_queries_and_label_independent_graph_support(self):
        a, p = training_fixture()
        _, _, graphs, folds = core.setup(a, p['core_parameters'], 'full', 'cpu')
        rng = np.random.default_rng(853)
        for f, ids in enumerate(folds):
            inner, meta = core.episode_indices(ids.numpy(), rng)
            self.assertEqual(np.intersect1d(inner, meta).size, 0)
            np.testing.assert_array_equal(np.sort(np.r_[inner, meta]), ids.numpy())
            self.assertTrue(np.all(a['folds'][np.r_[inner, meta]] == f))
            graph = graphs[f]
            queries = a['pairs_train'][:, ids.numpy()] + a['pair_offset'][:, None]
            n = int(a['node_counts'].sum())
            forbidden = np.r_[queries[0] * n + queries[1], queries[1] * n + queries[0]]
            self.assertEqual(np.intersect1d(graph[0][0].numpy() * n + graph[0][1].numpy(), forbidden).size, 0)
            changed = {key: value.copy() for key, value in a.items()}
            changed['y_train'][ids.numpy()] = 1 - changed['y_train'][ids.numpy()]
            _, _, alternative, _ = core.setup(changed, p['core_parameters'], 'full', 'cpu')
            for left, right in zip(graph, alternative[f]):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'training arrays only'):
            core.setup(dict(a, y_val=np.ones(3)), p['core_parameters'], 'full', 'cpu')

    def test_real_training_paths_budget_rng_and_fixed_input_boundary(self):
        a, p = training_fixture()
        initial, streams, episodes = [], [], []
        for kind in core.KINDS:
            with patch('torch.manual_seed', side_effect=AssertionError('all-device seeding forbidden')):
                warm, _, _, _ = experiment.fit(a, p, kind, .003, 917, 'cpu', fixed_epochs=1)
                model, value, trace, detail = experiment.fit(a, p, kind, .003, 917, 'cpu', fixed_epochs=7)
            initial.append(warm.backbone.state_dict())
            streams.append(model.training_final_state['cpu_rng'])
            episodes.append([r['episode_sha256'] for r in trace])
            self.assertIsNone(value)
            self.assertEqual(detail['gradient_evaluations'], 21)
            self.assertEqual(detail['task_optimizer_steps'], 7)
            self.assertEqual(detail['rescaler_optimizer_steps'], 7)
            self.assertEqual(detail['dropout_forward_passes'], 42)
            self.assertEqual(model.training_final_state['task_optimizer']['step'], 7)
            self.assertTrue(all(int(s['step']) == 7 for s in model.training_final_state['rescaler_optimizer']['state'].values()))
            self.assertTrue(all('anchor_composed' not in r for r in trace))
            if kind in ('full', 'no_outer_rescaling'):
                self.assertTrue(any(r['hypergradient_norm'] > 1e-10 for r in trace[1:]))
            else:
                self.assertTrue(all(r['hypergradient_norm'] == 0. for r in trace))
            if kind in ('full', 'no_meta_gradient'):
                self.assertTrue(any(r['direct_gradient_norm'] > 1e-6 for r in trace))
            else:
                self.assertTrue(all(r['direct_gradient_norm'] == 0. for r in trace))
            if kind == 'uniform':
                self.assertTrue(all(r['inner_weight_mean'] == 1. and r['meta_weight_mean'] == 1. for r in trace))
        for other in initial[1:]:
            for key, value in initial[0].items():
                torch.testing.assert_close(value, other[key], atol=0, rtol=0)
        for state in streams[1:]:
            torch.testing.assert_close(state, streams[0], atol=0, rtol=0)
        self.assertTrue(all(e == episodes[0] for e in episodes))
        with self.assertRaisesRegex(ValueError, 'training arrays only'):
            experiment.fit(dict(a, y_val=np.zeros(3)), p, 'full', .003, 917, 'cpu', fixed_epochs=7)

    def test_lnc_identity_requires_no_rescaler_or_training(self):
        a, p = training_fixture()
        keep = a['pairs_train'][0] < int(a['node_counts'][0])
        a['pairs_train'] = a['pairs_train'][:, keep]
        for key in ('y_train', 'folds', 'anchor_oof'):
            a[key] = a[key][keep]
        a['pair_offset'] = np.array([0, 14])
        for kind in core.KINDS:
            model, value, trace, detail = experiment.fit(a, p, kind, .003, 917, 'cpu', fixed_epochs=0)
            self.assertIsNone(value)
            self.assertFalse(model.state_dict())
            self.assertFalse(trace)
            self.assertFalse(detail['trained_neural_model'])
            self.assertEqual(detail['gradient_evaluations'], 0)
            self.assertIsNone(model.training_final_state)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
