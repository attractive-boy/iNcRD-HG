"""Independent covariance/transport mathematics and graph execution boundaries."""
from datetime import datetime, timezone
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import sqrtm
import torch

from improve import gaussian_transport_graph_v1 as core
from improve import anchor_gaussian_transport_experiment_v1 as experiment
from improve import tabular_foundation as base
from improve.test_nested_anchor_v2 import fixture

HERE = Path(__file__).resolve().parent/'research/distributional_geometry_20260916'
EVIDENCE = {}


def arrays():
    a, _ = fixture()
    a['anchor_oof'] = np.linspace(-.5, .5, len(a['y_train']))
    return a


def small_protocol():
    return dict(smoke=True, family='gaussian_transport', implementation_module='gaussian_transport_graph_v1',
        source_sha256={'gaussian_transport_graph_v1.py': base.digest(core.__file__)},
        core_parameters=dict(core.parameters(), width=8, heads=2, depth=1,
                             gaussian_heads=2, gaussian_dimension=3),
        kinds=list(core.KINDS), epochs=5, check_interval=5, weight_decay=.001, strong_weight=.1)


class GaussianTransportChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.random.default_generator.manual_seed(427)

    def test_square_root_scipy_and_repeated_spectrum_derivative(self):
        x = torch.randn(4, 3, 3, dtype=torch.float64)
        matrices = x@x.transpose(-1, -2)+.1*torch.eye(3)
        actual = core.positive_sqrt(matrices)
        expected = np.stack([sqrtm(a.numpy()) for a in matrices])
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-12, atol=2e-12)
        for a in (matrices[:1].requires_grad_(), torch.eye(3, dtype=torch.float64).requires_grad_()):
            self.assertTrue(torch.autograd.gradcheck(core.positive_sqrt, (a,), atol=1e-6, rtol=1e-4))
        with self.assertRaises(FloatingPointError):
            core.positive_sqrt(torch.diag(torch.tensor([1., 0.], dtype=torch.float64)))
        EVIDENCE['sqrt'] = dict(scipy_max_error=float(np.max(np.abs(actual.numpy()-expected))),
            gradcheck_random_spd=True, gradcheck_repeated_spectrum=True, no_eigenvalue_clipping=True)

    def test_empirical_moments_independent_numpy_and_translation(self):
        x = torch.randn(9, 2, 3, dtype=torch.float64, requires_grad=True)
        ids = torch.tensor([0, 2, 1, 1, 2, 1, 1, 2, 1])
        mu, cov = core.population_moments(x, ids, 3)
        expected_mu, expected_cov = [], []
        for i in range(3):
            values = x.detach().numpy()[ids.numpy() == i]
            expected_mu.append(values.mean(0))
            centered = values-values.mean(0)
            expected_cov.append(np.einsum('nhi,nhj->hij', centered, centered)/len(values))
        np.testing.assert_allclose(mu.detach(), expected_mu, atol=1e-14)
        np.testing.assert_allclose(cov.detach(), expected_cov, atol=1e-14)
        perm = torch.randperm(len(x))
        shifted, stable = core.population_moments(x[perm]+19., ids[perm], 3)
        torch.testing.assert_close(shifted, mu+19., atol=2e-14, rtol=2e-14)
        torch.testing.assert_close(stable, cov, atol=2e-14, rtol=2e-14)
        self.assertTrue(torch.autograd.gradcheck(lambda z: core.population_moments(z, ids, 3), (x,)))
        with self.assertRaises(ValueError):
            core.population_moments(x, ids, 4)
        EVIDENCE['moments'] = dict(numpy_population_covariance=True, singleton_covariance_zero=True,
            permutation_and_translation_invariance=True, gradcheck=True)

    def test_independent_two_dimensional_transport_and_gradients(self):
        a = torch.tensor([[[[2., .6], [.6, 1.]]]], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([[[[1., -.2], [-.2, 3.]]]], dtype=torch.float64, requires_grad=True)
        mu = torch.zeros(1, 1, 2, dtype=torch.float64)
        _, cost = core.gaussian_cost(mu, mu, a, b, 'full')
        aa, bb = a.detach().numpy()[0, 0], b.detach().numpy()[0, 0]
        independent = np.trace(aa)+np.trace(bb)-2*np.sqrt(np.trace(aa@bb)+2*np.sqrt(np.linalg.det(aa)*np.linalg.det(bb)))
        self.assertAlmostEqual(float(cost.detach()), float(independent), places=12)
        self.assertTrue(torch.autograd.gradcheck(lambda x, y: core.gaussian_cost(mu, mu, x, y, 'full')[1],
                                               (a, b), atol=1e-6, rtol=1e-4))
        _, reverse = core.gaussian_cost(mu, mu, b, a, 'full')
        torch.testing.assert_close(cost, reverse, atol=1e-12, rtol=1e-12)
        _, euclidean = core.gaussian_cost(mu, mu, a, b, 'root_euclidean')
        self.assertGreater(float(euclidean.detach()-cost.detach()), 1e-5)
        for kind in core.KINDS:
            _, zero = core.gaussian_cost(mu, mu, a, a, kind)
            self.assertLess(float(zero.detach().abs().max()), 1e-12)
        EVIDENCE['transport'] = dict(independent_2d_error=abs(float(cost.detach())-independent),
            covariance_gradcheck=True, symmetry=True, identity=True,
            noncommuting_root_euclidean_gap=float((euclidean-cost).detach()))

    def test_correlations_survive_and_common_rotations_preserve_full_distance(self):
        a = torch.tensor([[[[1., .7], [.7, 1.]]]], dtype=torch.float64)
        b = torch.tensor([[[[1., -.7], [-.7, 1.]]]], dtype=torch.float64)
        mu = torch.zeros(1, 1, 2, dtype=torch.float64)
        _, full = core.gaussian_cost(mu, mu, a, b, 'full')
        _, diagonal = core.gaussian_cost(mu, mu, a, b, 'diagonal')
        self.assertGreater(float(full), .1)
        self.assertEqual(float(diagonal), 0.)
        b = torch.tensor([[[[1., -.1], [-.1, 2.]]]], dtype=torch.float64)
        q = torch.tensor([[np.cos(.37), -np.sin(.37)], [np.sin(.37), np.cos(.37)]], dtype=torch.float64)
        for kind in ('full', 'root_euclidean'):
            cost = core.gaussian_cost(mu, mu, a, b, kind)[1]
            rotated = core.gaussian_cost(mu, mu, q@a@q.T, q@b@q.T, kind)[1]
            torch.testing.assert_close(cost, rotated, atol=1e-12, rtol=1e-12)
        EVIDENCE['information'] = dict(equal_mean_equal_diagonal_full_cost=float(full), diagonal_cost=0.,
            joint_orthogonal_invariance=True)

    def test_support_boundaries_node_permutation_and_query_batching(self):
        a, p = arrays(), small_protocol()
        model, xs, graphs, folds = core.setup(a, p['core_parameters'], 'full', 'cpu')
        with torch.no_grad():
            model.readout[-1].weight.normal_(0., .1)
        model.eval()
        pairs = torch.as_tensor(a['pairs_train']+a['pair_offset'][:, None])
        graph = graphs[0]
        y = model(xs, graph, pairs).detach()
        self.assertGreater(float(y.abs().max()), 1e-6)
        reversed_y = model(xs, graph, pairs.flip(1)).detach().flip(0)
        chunks = torch.cat([model(xs, graph, pairs[:, i:i+11]).detach() for i in range(0, pairs.shape[1], 11)])
        torch.testing.assert_close(y, reversed_y, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(y, chunks, atol=1e-6, rtol=1e-6)
        n = int(a['node_counts'].sum())
        for f, g in enumerate(graphs):
            q = pairs[:, folds[f]].numpy()
            forbidden = np.r_[q[0]*n+q[1], q[1]*n+q[0]]
            e = g[0].numpy()
            self.assertFalse(np.intersect1d(forbidden, e[0]*n+e[1]).size)
        altered = {k: v.copy() for k, v in a.items()}
        altered['y_train'] = 1-altered['y_train']
        _, _, altered_graphs, _ = core.setup(altered, p['core_parameters'], 'full', 'cpu')
        for g, h in zip(graphs, altered_graphs):
            for x, z in zip(g, h): torch.testing.assert_close(x, z, atol=0, rtol=0)
        old_to_new = np.arange(n)
        new_to_old, starts = [], np.r_[0, np.cumsum(a['node_counts'])]
        perm_xs = []
        for i, x in enumerate(xs):
            order = torch.randperm(len(x))
            perm_xs.append(x[order])
            old_to_new[starts[i]+order.numpy()] = starts[i]+np.arange(len(x))
            new_to_old.extend((starts[i]+order.numpy()).tolist())
        mapping = torch.as_tensor(old_to_new)
        transformed = (mapping[graph[0]], graph[1], graph[2][new_to_old])
        changed_y = model(perm_xs, transformed, mapping[pairs]).detach()
        torch.testing.assert_close(y, changed_y, atol=2e-6, rtol=2e-6)
        with self.assertRaisesRegex(ValueError, 'training arrays only'):
            core.setup(dict(a, y_val=np.zeros(1)), p['core_parameters'], 'full', 'cpu')
        EVIDENCE['boundary'] = dict(forbidden_query_edges=0, query_label_invariance=True,
            within_type_node_permutation_error=float((y-changed_y).abs().max()),
            batching_error=float((y-chunks).abs().max()), nonzero_decoder=True)

    def test_matched_training_identity_and_every_parameter_gradient(self):
        a, p = arrays(), small_protocol()
        reference, rng = None, None
        final_rngs = []
        for kind in core.KINDS:
            torch.random.default_generator.manual_seed(716)
            initial, _, _, _ = core.setup(a, p['core_parameters'], kind, 'cpu')
            if reference is None:
                reference, rng = experiment.cpu_copy(initial.state_dict()), torch.random.get_rng_state()
            else:
                for name, value in initial.state_dict().items():
                    torch.testing.assert_close(value, reference[name], atol=0, rtol=0)
                torch.testing.assert_close(torch.random.get_rng_state(), rng, atol=0, rtol=0)
            model, value, trace, detail = experiment.fit(a, p, kind, .003, 716, 'cpu', fixed_epochs=5)
            self.assertIsNone(value)
            self.assertFalse(any('anchor_composed' in row for row in trace))
            self.assertEqual(detail['optimizer_steps'], 5)
            self.assertEqual(detail['training_steps'], 5)
            for name, value in model.named_parameters():
                self.assertIsNotNone(value.grad, name)
                self.assertTrue(torch.isfinite(value.grad).all(), name)
            final = model.training_final_state
            final_rngs.append(final['cpu_rng'])
            self.assertEqual(final['steps'], 5)
            for value in final['optimizer']['state'].values(): self.assertEqual(int(value['step']), 5)
        for r in final_rngs[1:]: torch.testing.assert_close(r, final_rngs[0], atol=0, rtol=0)
        lnc = arrays()
        lnc['pair_offset'] = np.array([0, 14])
        for kind in core.KINDS:
            model, value, trace, detail = experiment.fit(lnc, p, kind, .003, 716, 'cpu', fixed_epochs=5)
            self.assertFalse(detail['trained_neural_model'])
            self.assertEqual(detail['training_steps'], 0)
            self.assertFalse(list(model.parameters()))
            self.assertFalse(trace)
            result = model([], None, torch.tensor([[0, 1], [14, 15]]))
            torch.testing.assert_close(result, torch.zeros_like(result), atol=0, rtol=0)
        EVIDENCE['training'] = dict(initial_states_equal=True, random_streams_equal=True,
            every_parameter_finite_gradient=True, identity_lnc_no_training=True, fixed_fit_has_no_evaluation=True)

    def test_partial_numerical_failure_preserves_exact_completed_steps(self):
        a, p = arrays(), small_protocol()
        original, count = core.GaussianTransportGraph.forward, [0]
        def broken(model, *args):
            count[0] += 1
            if count[0] == 3: raise FloatingPointError('injected SPD failure')
            return original(model, *args)
        with patch.object(core.GaussianTransportGraph, 'forward', broken):
            with self.assertRaises(FloatingPointError) as caught:
                experiment.fit(a, p, 'full', .003, 716, 'cpu', fixed_epochs=5)
        error = caught.exception
        self.assertEqual(error.training_partial_state['steps'], 2)
        self.assertEqual(error.training_failure_context['operation'], 'forward')
        self.assertEqual(error.training_failure_context['current_step'], 3)
        for value in error.training_partial_state['optimizer']['state'].values():
            self.assertEqual(int(value['step']), 2)
        with self.assertRaisesRegex(ValueError, 'training arrays only'):
            experiment.fit(dict(a, y_val=np.zeros(2)), p, 'full', .003, 716, 'cpu', fixed_epochs=5)
        EVIDENCE['failure'] = dict(injected_step=3, completed_steps=2, exact_optimizer_snapshot=True)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GaussianTransportChecks))
    path = HERE/'GAUSSIAN_TRANSPORT_UNIT_V1.json'
    if path.exists(): raise ValueError('Use a new version for revised test evidence')
    base.write_json(path, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        success=result.wasSuccessful(), tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        evidence=EVIDENCE, no_performance_metrics=True,
        source_sha256={str(Path(p).resolve()):base.digest(p) for p in
            [__file__, core.__file__, experiment.__file__, Path(__file__).with_name('test_nested_anchor_v2.py')]})))
    raise SystemExit(0 if result.wasSuccessful() else 1)
