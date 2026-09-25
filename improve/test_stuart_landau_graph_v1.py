"""Independent cubic, Cartesian derivatives, pinned-author and graph audits."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import brentq
import torch
from torch.nn import functional as F

from improve import stuart_landau_graph_v1 as core
from improve import anchor_stuart_landau_experiment_v1 as experiment
from improve import tabular_foundation as base
from improve.test_nested_anchor_v2 import fixture

HERE = Path(__file__).resolve().parent / 'research/oscillatory_next_20260916'
EVIDENCE = {}


def arrays():
    a, _ = fixture()
    a['anchor_oof'] = np.linspace(-.5, .5, len(a['y_train']))
    return a


def small_protocol():
    parameters = dict(core.parameters(), width=4, depth=3, readout_width=8)
    return dict(smoke=True, family='stuart_landau', implementation_module='stuart_landau_graph_v1',
        source_sha256={'stuart_landau_graph_v1.py': base.digest(core.__file__)},
        core_parameters=parameters, kinds=list(core.KINDS), epochs=5, check_interval=5,
        weight_decay=.001, strong_weight=.1)


def author_solver():
    records = base.read_json(HERE / 'SLGNN_AUTHOR_SOURCE_V1.json')
    row = next(r for r in records if r['path'] == 'src/slmodel.py')
    path = experiment.ROOT.parent / row['file']
    assert base.digest(path) == row['sha256'] and row['git_blob_verified']
    assert row['commit'] == '40944b1e0d2b954de6b46f31b6a9a6f3dc04f095'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SLGCN')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ('solve_implicit_r', 'ode_step')]
    assert len(methods) == 2
    selected = ast.Module(body=[ast.ClassDef(name='SLGCN', bases=[], keywords=[], body=methods,
        decorator_list=[], type_params=[])], type_ignores=[])
    namespace = {'torch': torch}
    exec(compile(ast.fix_missing_locations(selected), str(path), 'exec'), namespace)
    instance = namespace['SLGCN']()
    p = core.parameters()
    instance.dt, instance.p, instance.k, instance.tol = p['dt'], 2, p['coupling_strength'], 1e-13
    instance.zeta, instance.nu = complex(p['alpha'], p['omega']), complex(p['beta'], p['gamma'])
    instance.implicit_r_tol_break = False
    instance.activation = lambda x: F.leaky_relu(x, p['leaky_relu_slope'])
    return instance, row


class OscillatorChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.random.default_generator.manual_seed(849)

    def test_positive_cubic_against_scipy_and_implicit_gradients(self):
        errors = []
        for a in (.2, .9, 1.7):
            for s in (1e-9, .1, 2.):
                values = np.array([0., 1e-24, 1e-12, .01, 1., 100., 1e6])
                v = torch.tensor(values, dtype=torch.float64)
                q = core.SquaredRadialRoot.apply(v, v.new_tensor(a), v.new_tensor(s))
                reference = []
                for value in values:
                    r = 0. if value == 0 else brentq(lambda radius: s*radius**3+a*radius-np.sqrt(value),
                        0., np.sqrt(value)/a+1e-8, xtol=1e-300, rtol=1e-14)
                    reference.append(r*r)
                relative = np.max(abs(q.numpy()-reference)/(1.+np.asarray(reference)))
                self.assertLess(relative, 1e-12)
                self.assertTrue(torch.all(q >= 0))
                errors.append(float(relative))
        v = torch.tensor([.03, .7, 3.2], dtype=torch.float64, requires_grad=True)
        a = torch.tensor(.9, dtype=torch.float64, requires_grad=True)
        s = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(core.SquaredRadialRoot.apply, (v, a, s),
            eps=1e-6, atol=2e-6, rtol=1e-5))
        EVIDENCE['root'] = dict(independent_scipy_cases=len(errors)*7,
            maximum_scaled_root_error=max(errors), all_positive_or_zero=True,
            implicit_derivatives_for_squared_input_and_both_coefficients=True)

    def test_author_step_gradients_and_zero_radius_extension(self):
        author, row = author_solver()
        p = core.parameters()
        state = torch.tensor([[.02, -.01], [.3, .7], [-1.2, .5], [2.1, -1.1]],
                             dtype=torch.float64, requires_grad=True)
        def original(x):
            z = torch.view_as_complex(x.contiguous())
            return torch.view_as_real(author.ode_step(z, None, lambda value, edges: torch.zeros_like(value)))
        actual, _ = core.local_step(state, p, 'full')
        expected = original(state)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        grad_a = torch.autograd.grad(actual.square().sum(), state, retain_graph=True)[0]
        grad_b = torch.autograd.grad(expected.square().sum(), state)[0]
        torch.testing.assert_close(grad_a, grad_b, atol=1e-11, rtol=1e-11)
        zero = torch.zeros(2, dtype=torch.float64)
        local_jacobian = torch.autograd.functional.jacobian(lambda x: core.local_step(x, p, 'full')[0], zero)
        author_jacobian = torch.autograd.functional.jacobian(original, zero)
        angle, scale = p['dt']*p['omega'], 1./(1.-p['dt']*p['alpha'])
        analytic = scale*torch.tensor([[np.cos(angle), -np.sin(angle)],
                                      [np.sin(angle), np.cos(angle)]], dtype=torch.float64)
        torch.testing.assert_close(local_jacobian, analytic, atol=1e-13, rtol=1e-13)
        finite_difference = torch.stack([(original(zero+1e-6*axis)-original(zero-1e-6*axis))/(2e-6)
                                         for axis in torch.eye(2, dtype=torch.float64)], dim=1)
        torch.testing.assert_close(local_jacobian, finite_difference, atol=1e-11, rtol=1e-11)
        source_zero_error = float((author_jacobian-analytic).abs().max())
        self.assertGreater(source_zero_error, .1)
        zero_and_nonzero = torch.cat((torch.zeros(1, 2, dtype=torch.float64), state.detach())).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda x: core.local_step(x, p, 'full')[0],
            (zero_and_nonzero,), eps=1e-6, atol=2e-6, rtol=1e-5))
        EVIDENCE['author'] = dict(commit=row['commit'], source_sha256=row['sha256'],
            selected_methods=['SLGCN.solve_implicit_r', 'SLGCN.ode_step'], selected_method_bodies_modified=False,
            full_author_graph_model_executed=False, maximum_forward_error=float((actual-expected).detach().abs().max()),
            maximum_nonzero_gradient_error=float((grad_a-grad_b).abs().max()),
            author_zero_radius_autograd_error=source_zero_error, author_zero_radius_gradient_claimed_passing=False,
            local_zero_jacobian_error=float((local_jacobian-analytic).abs().max()),
            local_matches_author_central_difference_at_zero=True, cartesian_gradcheck_includes_zero=True)

    def test_complex_coupling_against_dense_numpy_and_gradients(self):
        n, width = 4, 3
        send, recv = np.indices((n, n)).reshape(2, -1)
        edges = torch.tensor(np.stack((send, recv)))
        relation = torch.tensor((send+2*recv) % 3)
        context = core.pack_graph((edges, relation, torch.zeros(n, 1)))
        layer = core.ComplexTypedGCN(width, 3, .2).double()
        with torch.no_grad():
            layer.imaginary_weight.normal_(std=.1)
            layer.relation_gate.normal_(std=.2)
        state = torch.randn(n, width, 2, dtype=torch.float64, requires_grad=True)
        complex_state = state.detach().numpy()[..., 0] + 1j*state.detach().numpy()[..., 1]
        matrix = layer.real_weight.detach().numpy() + 1j*layer.imaginary_weight.detach().numpy()
        projected = complex_state @ matrix.T
        gates = 2./(1.+np.exp(-layer.relation_gate.detach().numpy()))
        reference = np.zeros((n, width), dtype=np.complex128)
        for sender, receiver, kind in zip(send, recv, relation.numpy()):
            reference[receiver] += projected[sender]*gates[kind]/4.
        reference = np.stack((np.where(reference.real >= 0, reference.real, .2*reference.real),
                              np.where(reference.imag >= 0, reference.imag, .2*reference.imag)), axis=-1)
        actual = layer(state, context)
        np.testing.assert_allclose(actual.detach().numpy(), reference, atol=1e-12, rtol=1e-12)
        names, values = zip(*layer.named_parameters())
        self.assertTrue(torch.autograd.gradcheck(lambda x, *v: torch.func.functional_call(
            layer, dict(zip(names, v)), (x, context)), (state,)+values, eps=1e-6, atol=1e-6, rtol=1e-4))
        author, _ = author_solver()
        def convolution(z, unused):
            transformed = z @ torch.complex(layer.real_weight, layer.imaginary_weight).T
            messages = transformed[edges[0]]*(2.*layer.relation_gate[relation].sigmoid())*.25
            return torch.zeros_like(z).index_add(0, edges[1], messages)
        z = torch.view_as_complex(state.contiguous())
        source_step = torch.view_as_real(author.ode_step(z, edges, convolution))
        local = core.local_step(state, core.parameters(), 'full')[0] + .1*actual
        torch.testing.assert_close(source_step, local, atol=1e-12, rtol=1e-12)
        EVIDENCE['coupling'] = dict(independent_numpy_max_error=float(abs(actual.detach().numpy()-reference).max()),
            all_active_complex_map_and_gate_gradients_checked=True, full_author_order_step_error=float(
                (local-source_step).detach().abs().max()), author_pseudocode_order_not_substituted=True)

    def test_control_limits_rotation_and_continuous_solution(self):
        p = core.parameters()
        x = torch.tensor([[.2, 0.], [1.3, 0.]], dtype=torch.float64)
        errors = []
        r0 = x[:, 0].numpy()
        final_time = 1.
        radius = 1./np.sqrt(1.+(1./r0**2-1.)*np.exp(-2.*final_time))
        integral = .5*np.log1p(r0**2*np.expm1(2.*final_time))
        angle = final_time-.5*integral
        exact = np.stack((radius*np.cos(angle), radius*np.sin(angle)), axis=-1)
        for steps in (20, 40):
            state = x.clone()
            for _ in range(steps):
                state, _ = core.local_step(state, dict(p, dt=final_time/steps), 'full')
            errors.append(float(abs(state.numpy()-exact).max()))
        self.assertLess(errors[1], .6*errors[0])
        off = dict(p, gamma=0.)
        torch.testing.assert_close(core.local_step(x, off, 'full')[0],
            core.local_step(x, off, 'no_phase_feedback')[0], atol=0., rtol=0.)
        identity = core.local_step(x, dict(p, alpha=0., omega=0.), 'linear_oscillator')[0]
        torch.testing.assert_close(identity, x, atol=0., rtol=0.)
        rotation = torch.tensor([[np.cos(.7), -np.sin(.7)], [np.sin(.7), np.cos(.7)]], dtype=torch.float64)
        base_step = core.local_step(x, p, 'full')[0]
        rotated = core.local_step(x @ rotation.T, p, 'full')[0]
        torch.testing.assert_close(rotated, base_step @ rotation.T, atol=1e-13, rtol=1e-13)
        EVIDENCE['limits'] = dict(uncoupled_exact_solution_errors=errors, first_order_time_convergence=True,
            no_feedback_equals_full_at_gamma_zero=True, linear_zero_growth_zero_rotation_identity=True,
            local_global_phase_rotation_equivariance=True, full_GCN_phase_equivariance_claimed=False)

    def test_graph_boundary_permutation_and_query_order(self):
        a, p = arrays(), small_protocol()
        with self.assertRaisesRegex(ValueError, 'training arrays only'):
            core.setup(dict(a, y_val=np.array([0.])), p['core_parameters'], 'full', 'cpu')
        model, xs, graphs, folds = core.setup(a, p['core_parameters'], 'full', 'cpu')
        graph, n = graphs[0], int(a['node_counts'].sum())
        query = torch.as_tensor(a['pairs_train'][:, folds[0]]+a['pair_offset'][:, None])
        codes = graph.physical[0][0].numpy()*n+graph.physical[0][1].numpy()
        forbidden = np.r_[query[0].numpy()*n+query[1].numpy(), query[1].numpy()*n+query[0].numpy()]
        self.assertEqual(np.intersect1d(codes, forbidden).size, 0)
        changed = {k: v.copy() for k, v in a.items()}
        changed['y_train'][folds[0].numpy()] = 1.-changed['y_train'][folds[0].numpy()]
        _, _, alternate, _ = core.setup(changed, p['core_parameters'], 'full', 'cpu')
        for current, expected in zip(graph.physical, alternate[0].physical):
            torch.testing.assert_close(current, expected, atol=0., rtol=0.)
        torch.testing.assert_close(graph.normalization, alternate[0].normalization, atol=0., rtol=0.)
        permutation, changed_xs, start = [], [], 0
        for x in xs:
            order = torch.randperm(len(x)); new = torch.empty_like(x); new[order] = x
            changed_xs.append(new); permutation.append(order+start); start += len(x)
        permutation = torch.cat(permutation)
        counts = torch.empty_like(graph.physical[2]); counts[permutation] = graph.physical[2]
        mapped = core.pack_graph((permutation[graph.physical[0]], graph.physical[1], counts))
        model.eval()
        with torch.no_grad():
            model.readout[-1].weight.normal_()
            expected = model(xs, graph, query)
            reordered = model(changed_xs, mapped, permutation[query])
            reverse = model(xs, graph, query.flip(1)).flip(0)
        for actual in (reordered, reverse):
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
        EVIDENCE['graph'] = dict(training_only_setup=True, query_edges_and_reverse_absent=True,
            omitted_labels_cannot_change_normalized_support=True,
            node_permutation_max_error=float((reordered-expected).abs().max()),
            query_order_max_error=float((reverse-expected).abs().max()))

    def test_executor_initialization_gradients_identity_and_partial_state(self):
        a, p, rngs, counts = arrays(), small_protocol(), [], []
        with self.assertRaisesRegex(ValueError, 'training arrays only'):
            experiment.fit(dict(a, y_val=np.array([0.])), p, 'full', .001, 991, 'cpu', fixed_epochs=5)
        with self.assertRaisesRegex(ValueError, 'cuda:1'):
            experiment.fit(a, p, 'full', .001, 991, 'cuda:0', fixed_epochs=5)
        initial_state, initial_rng = None, None
        for kind in core.KINDS:
            torch.random.default_generator.manual_seed(991)
            initial, xs, graphs, folds = experiment.setup(a, p, kind, 'cpu')
            if initial_state is None:
                initial_state, initial_rng = experiment.cpu_copy(initial.state_dict()), torch.random.get_rng_state()
            else:
                for name, value in initial.state_dict().items():
                    torch.testing.assert_close(value, initial_state[name], atol=0., rtol=0.)
                torch.testing.assert_close(torch.random.get_rng_state(), initial_rng, atol=0., rtol=0.)
            initial.eval()
            query = torch.as_tensor(a['pairs_train'][:, folds[0]]+a['pair_offset'][:, None])
            torch.testing.assert_close(initial(xs, graphs[0], query), torch.zeros(query.shape[1]), atol=0., rtol=0.)
            model, value, trace, detail = experiment.fit(a, p, kind, .001, 991, 'cpu', fixed_epochs=5)
            self.assertIsNone(value)
            self.assertEqual(detail['optimizer_steps'], 5)
            d = detail['oscillator_diagnostics']
            self.assertEqual(d['calls'], 5)
            self.assertEqual(d['local_steps'], 15)
            self.assertEqual(d['radial_solves'], 15)
            self.assertLessEqual(d['maximum_radial_relative_residual'], p['core_parameters']['radial_residual_tolerance'])
            self.assertTrue(all('anchor_composed' not in row for row in trace))
            self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters()))
            final = model.training_final_state
            self.assertFalse(final['frozen_parameter_names'])
            self.assertEqual(len(final['optimizer_parameter_names']), len(final['optimizer']['state']))
            self.assertTrue(all(int(v['step']) == 5 for v in final['optimizer']['state'].values()))
            rngs.append(final['cpu_rng'])
            counts.append({key: d[key] for key in ('calls', 'local_steps', 'radial_solves', 'solver_entries', 'coupling_messages')})
        for rng in rngs[1:]:
            torch.testing.assert_close(rng, rngs[0], atol=0., rtol=0.)
        self.assertTrue(all(v == counts[0] for v in counts))
        ln = dict(a, pair_offset=np.array([0, 14]))
        identity, _, trace, detail = experiment.fit(ln, p, 'full', .001, 991, 'cpu', fixed_epochs=5)
        self.assertEqual(identity.state_dict(), {})
        self.assertEqual(trace, [])
        self.assertEqual(detail['optimizer_steps'], 0)
        original = core.StuartLandauGraph.forward
        def fail_third(model, *args):
            if model.oscillator_diagnostics['calls'] == 2:
                raise FloatingPointError('Synthetic registered oscillator failure')
            return original(model, *args)
        with patch.object(core.StuartLandauGraph, 'forward', fail_third):
            with self.assertRaisesRegex(FloatingPointError, 'Synthetic registered oscillator failure') as caught:
                experiment.fit(a, p, 'full', .001, 111, 'cpu', fixed_epochs=5)
        self.assertEqual(caught.exception.training_partial_state['steps'], 2)
        self.assertEqual(caught.exception.training_failure_context['current_step'], 3)
        EVIDENCE['executor'] = dict(initial_weights_and_random_streams_exactly_equal=True,
            zero_initial_residual=True, final_rng_equal=True, all_parameter_gradients_finite=True,
            computational_counts_equal=True, counts=counts[0], identity_without_training=True,
            exact_partial_failure_state_verified=True, formal_fits=0, evaluation_labels_read=False)


if __name__ == '__main__':
    destination = HERE / 'STUART_LANDAU_UNIT_V1.json'
    if destination.exists():
        raise ValueError('Preserve existing unit evidence')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OscillatorChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sources = [Path(__file__), Path(core.__file__), Path(experiment.__file__),
        experiment.ROOT / 'test_nested_anchor_v2.py', experiment.ROOT / 'overlap_block_models.py',
        experiment.ROOT / 'sharpness_graph_v1.py', HERE / 'SLGNN_AUTHOR_SOURCE_V1.json',
        HERE / 'slgnn_author_v1_src__slmodel.py']
    record = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        tests=result.testsRun, success=result.wasSuccessful(), failures=len(result.failures),
        errors=len(result.errors), evidence=EVIDENCE,
        source_sha256={str(path): base.digest(path) for path in sources},
        formal_neural_fits=0, evaluation_labels_read=False))
    base.write_json(destination, record)
    raise SystemExit(0 if result.wasSuccessful() else 1)
