"""Pinned-author, independent polynomial, derivative, and training-boundary checks."""
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from improve import spectral_triadic_graph_v1 as core
from improve import anchor_spectral_triadic_experiment_v1 as experiment
from improve import tabular_foundation as base
from improve.test_nested_anchor_v2 import fixture

HERE = Path(__file__).resolve().parent / 'research/tensor_interaction_next_20260916'
EVIDENCE = {}


def arrays():
    a, _ = fixture()
    a['anchor_oof'] = np.linspace(-.5, .5, len(a['y_train']))
    return a


def small_protocol():
    p = dict(core.parameters(), width=8, heads=2)
    return dict(smoke=True, family='spectral_triadic', implementation_module='spectral_triadic_graph_v1',
        source_sha256={'spectral_triadic_graph_v1.py': base.digest(core.__file__)},
        core_parameters=p, kinds=list(core.KINDS), epochs=5, check_interval=5,
        weight_decay=.001, strong_weight=.1)


def author_layer():
    records = base.read_json(HERE / 'SHONN_AUTHOR_SOURCE_V1.json')
    row = next(r for r in records if r['path'] == 'lib/layers.py')
    path = experiment.ROOT.parent / row['file']
    assert base.digest(path) == row['sha256'] and row['git_blob_verified']
    assert row['commit'] == 'bf38adfcfc441d4bf71e9eff43d87e41a0f1e7a7'
    spec = importlib.util.spec_from_file_location('verified_shonn_layers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SpectralTriadic, row


class SpectralTriadicChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.random.default_generator.manual_seed(852)

    def test_author_initialization_forward_and_parameter_gradients(self):
        author, row = author_layer()
        prior_dtype = torch.get_default_dtype()
        errors = []
        try:
            torch.set_default_dtype(torch.float64)
            for bias in (False, True):
                torch.random.default_generator.manual_seed(43)
                expected = author(4, 3, bias=bias)
                author_rng = torch.random.get_rng_state()
                torch.random.default_generator.manual_seed(43)
                actual = core.SpectralTriadic(4, 3, bias=bias)
                torch.testing.assert_close(torch.random.get_rng_state(), author_rng, atol=0., rtol=0.)
                for name, value in expected.state_dict().items():
                    torch.testing.assert_close(actual.state_dict()[name], value, atol=0., rtol=0.)
                x = torch.randn(5, 4, requires_grad=True)
                y, reference = actual(x), expected(x)
                torch.testing.assert_close(y, reference, atol=1e-13, rtol=1e-13)
                named_a = [(name, value) for name, value in actual.named_parameters() if value.requires_grad]
                named_b = dict(expected.named_parameters())
                grad_a = torch.autograd.grad(y.square().sum(), [x] + [v for _, v in named_a], retain_graph=True)
                grad_b = torch.autograd.grad(reference.square().sum(), [x] + [named_b[n] for n, _ in named_a])
                for left, right in zip(grad_a, grad_b):
                    torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
                errors.append(dict(bias=bias, forward=float((y-reference).detach().abs().max()),
                    gradient=max(float((left-right).abs().max()) for left, right in zip(grad_a, grad_b))))
        finally:
            torch.set_default_dtype(prior_dtype)
        EVIDENCE['author'] = dict(commit=row['commit'], source_sha256=row['sha256'],
            full_unmodified_layer_module_executed=True, original_training_scripts_executed=False,
            initialization_and_rng_exact=True, errors=errors)

    def test_independent_polynomial_hessian_and_gradcheck(self):
        model = core.SpectralTriadic(3, 2).double()
        x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        state = {k: v.detach().numpy() for k, v in model.state_dict().items()}
        reference = np.zeros((2, 2))
        xx = x.detach().numpy()
        for batch in range(2):
            for k in range(2):
                for i in range(4):
                    value = xx[batch, i] if i < 3 else 1.
                    reference[batch, k] += (state['l_in'][i]-state['l_out'][k])*state['phi'][k, i]*value
                index = 0
                for i in range(3):
                    for j in range(i, 3):
                        reference[batch, k] += ((state['l_in_triadic'][index]-state['l_out_triadic'][k])
                            * state['phi_triadic'][k, index] * xx[batch, i] * xx[batch, j])
                        index += 1
        actual = model(x)
        np.testing.assert_allclose(actual.detach().numpy(), reference, atol=1e-13, rtol=1e-13)
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        values = tuple(dict(model.named_parameters())[n] for n in names)
        def call(inputs, *parameters):
            return torch.func.functional_call(model, dict(zip(names, parameters)), (inputs,))
        self.assertTrue(torch.autograd.gradcheck(call, (x,) + values, eps=1e-6, atol=1e-6, rtol=1e-4))
        hessian = torch.autograd.functional.hessian(lambda v: model(v[None])[0, 0], x[0])
        expected_hessian = torch.zeros(3, 3, dtype=torch.float64)
        weights = model.quadratic_weight().detach()[0]
        for index, (i, j) in enumerate(zip(model.triu_rows, model.triu_cols)):
            expected_hessian[i, j] = (2. if i == j else 1.) * weights[index]
            expected_hessian[j, i] = expected_hessian[i, j]
        torch.testing.assert_close(hessian, expected_hessian, atol=1e-13, rtol=1e-13)
        EVIDENCE['polynomial'] = dict(independent_numpy_max_error=float(abs(actual.detach().numpy()-reference).max()),
            input_and_all_active_parameter_gradcheck=True,
            analytic_hessian_max_error=float((hessian-expected_hessian).abs().max()),
            upper_triangular_cross_terms_counted_once=True, square_hessian_factor_two_verified=True)

    def test_controls_gauge_sharing_and_parameter_counts(self):
        prior_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            modules, rngs = {}, []
            for kind in core.KINDS:
                torch.random.default_generator.manual_seed(93)
                modules[kind] = core.SpectralTriadic(4, 3, kind)
                rngs.append(torch.random.get_rng_state())
            for rng in rngs[1:]:
                torch.testing.assert_close(rng, rngs[0], atol=0., rtol=0.)
            x = torch.randn(3, 4, requires_grad=True)
            full, linear, direct = (modules[k] for k in core.KINDS)
            y, z = full(x), direct(x)
            torch.testing.assert_close(y, z, atol=0., rtol=0.)
            torch.testing.assert_close(torch.autograd.grad(y.sum(), x)[0],
                torch.autograd.grad(z.sum(), x)[0], atol=0., rtol=0.)
            h0 = torch.autograd.functional.hessian(lambda v: linear(v[None]).sum(), x[0])
            torch.testing.assert_close(h0, torch.zeros_like(h0), atol=0., rtol=0.)
            normalized = full.quadratic_weight() / full.phi_triadic
            rectangles = normalized - normalized[:1] - normalized[:, :1] + normalized[0, 0]
            torch.testing.assert_close(rectangles, torch.zeros_like(rectangles), atol=1e-14, rtol=1e-14)
            reference = full(x).detach()
            with torch.no_grad():
                full.l_in_triadic.add_(.731)
                full.l_out_triadic.add_(.731)
            torch.testing.assert_close(full(x), reference, atol=1e-13, rtol=1e-13)
            with torch.no_grad():
                direct.phi_triadic[1, 1].add_(.5)
            ratio = direct.phi_triadic / full.phi_triadic
            witness = ratio[0, 0] + ratio[1, 1] - ratio[0, 1] - ratio[1, 0]
            self.assertGreater(float(witness.detach().abs()), .01)
        finally:
            torch.set_default_dtype(prior_dtype)
        counts = []
        for n, k in ((16, 16), (32, 32), (128, 64), (64, 64)):
            pairs = n*(n+1)//2
            full = core.SpectralTriadic(n, k, 'full')
            direct = core.SpectralTriadic(n, k, 'direct_quadratic')
            full_count = k*(n+1)+(n+1)+k+pairs+k
            direct_count = k*(n+1)+(n+1)+k+k*pairs
            self.assertEqual(sum(p.numel() for p in full.parameters() if p.requires_grad), full_count)
            self.assertEqual(sum(p.numel() for p in direct.parameters() if p.requires_grad), direct_count)
            counts.append(dict(inputs=n, outputs=k, full_trainable=full_count, direct_trainable=direct_count,
                fixed_tensor_elements=k*pairs, quadratic_multiply_adds_per_query=k*pairs))
        EVIDENCE['controls'] = dict(initial_rng_equal=True, full_direct_initial_function_and_input_gradient_equal=True,
            linear_layer_hessian_exact_zero=True, normalized_tensor_additive_constraint=True,
            spectral_gauge_shift_invariance=True, direct_tensor_can_violate_constraint=True,
            equal_trainable_capacity_claimed=False, complexity=counts)

    def test_graph_query_boundary_and_node_permutation(self):
        a, p = arrays(), small_protocol()
        with self.assertRaisesRegex(ValueError, 'training arrays only'):
            core.setup(dict(a, y_val=np.array([0.])), p['core_parameters'], 'full', 'cpu')
        model, xs, graphs, folds = core.setup(a, p['core_parameters'], 'full', 'cpu')
        graph, n = graphs[0], int(a['node_counts'].sum())
        query = torch.as_tensor(a['pairs_train'][:, folds[0]] + a['pair_offset'][:, None])
        codes = graph[0][0].numpy()*n + graph[0][1].numpy()
        forbidden = np.r_[query[0].numpy()*n+query[1].numpy(), query[1].numpy()*n+query[0].numpy()]
        self.assertEqual(np.intersect1d(codes, forbidden).size, 0)
        changed = {k: v.copy() for k, v in a.items()}
        changed['y_train'][folds[0].numpy()] = 1. - changed['y_train'][folds[0].numpy()]
        _, _, alternate, _ = core.setup(changed, p['core_parameters'], 'full', 'cpu')
        for current, expected in zip(graph, alternate[0]):
            torch.testing.assert_close(current, expected, atol=0., rtol=0.)
        permutation, changed_xs, start = [], [], 0
        for x in xs:
            order = torch.randperm(len(x))
            new_x = torch.empty_like(x)
            new_x[order] = x
            changed_xs.append(new_x)
            permutation.append(order + start)
            start += len(x)
        permutation = torch.cat(permutation)
        counts = torch.empty_like(graph[2]); counts[permutation] = graph[2]
        mapped = (permutation[graph[0]], graph[1], counts)
        model.eval()
        with torch.no_grad():
            model.readout.weight.normal_()
            reference = model(xs, graph, query)
            reordered = model(changed_xs, mapped, permutation[query])
            reverse = model(xs, graph, query.flip(1)).flip(0)
        for actual in (reordered, reverse):
            torch.testing.assert_close(actual, reference, atol=3e-6, rtol=3e-6)
        EVIDENCE['graph'] = dict(query_edges_and_reverse_absent=True, omitted_labels_do_not_affect_graph=True,
            node_permutation_max_error=float((reference-reordered).abs().max()),
            query_order_max_error=float((reference-reverse).abs().max()))

    def test_executor_identity_gradient_masks_and_random_streams(self):
        a, p, rngs, records = arrays(), small_protocol(), [], []
        with self.assertRaisesRegex(ValueError, 'training arrays only'):
            experiment.fit(dict(a, y_val=np.array([0.])), p, 'full', .001, 994, 'cpu', fixed_epochs=5)
        with self.assertRaisesRegex(ValueError, 'cuda:1'):
            experiment.fit(a, p, 'full', .001, 994, 'cuda:0', fixed_epochs=5)
        for kind in core.KINDS:
            torch.random.default_generator.manual_seed(994)
            initial, _, _, _ = experiment.setup(a, p, kind, 'cpu')
            model, value, trace, detail = experiment.fit(a, p, kind, .001, 994, 'cpu', fixed_epochs=5)
            self.assertIsNone(value)
            self.assertEqual(detail['optimizer_steps'], 5)
            self.assertEqual(detail['triadic_diagnostics']['calls'], 5)
            self.assertTrue(all('anchor_composed' not in row for row in trace))
            final = model.training_final_state
            rngs.append(final['cpu_rng'])
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                else:
                    self.assertIsNone(parameter.grad)
                    torch.testing.assert_close(parameter.detach(), initial.state_dict()[name], atol=0., rtol=0.)
            ids = final['optimizer']['param_groups'][0]['params']
            for name, index in zip(final['optimizer_parameter_names'], ids):
                state = final['optimizer']['state'][index]
                self.assertEqual(int(state['step']), 5)
                if kind == 'linear_only' and 'triadic' in name:
                    self.assertEqual(int(torch.count_nonzero(state['exp_avg'])), 0)
                    self.assertEqual(int(torch.count_nonzero(state['exp_avg_sq'])), 0)
            if kind == 'full':
                self.assertGreater(float(model.pair_layers[0].l_in_triadic.grad.abs().sum()), 1e-10)
            if kind == 'direct_quadratic':
                self.assertGreater(float(model.pair_layers[0].phi_triadic.grad.abs().sum()), 1e-10)
            records.append(dict(kind=kind, trainable_parameters=detail['trainable_parameters'],
                fixed_parameters=detail['fixed_parameters'], diagnostics=detail['triadic_diagnostics']))
        for rng in rngs[1:]:
            torch.testing.assert_close(rng, rngs[0], atol=0., rtol=0.)
        ln = dict(a, pair_offset=np.array([0, 14]))
        identity, value, trace, detail = experiment.fit(ln, p, 'full', .001, 994, 'cpu', fixed_epochs=5)
        self.assertEqual(identity.state_dict(), {})
        self.assertEqual(trace, [])
        self.assertEqual(detail['optimizer_steps'], 0)
        query = torch.tensor([[0, 1], [14, 15]])
        torch.testing.assert_close(identity([], None, query), torch.zeros(2), atol=0., rtol=0.)
        EVIDENCE['executor'] = dict(training_only_boundary=True, cuda0_rejected=True,
            evaluation_metrics_computed=False, training_random_streams_equal=True,
            all_active_gradients_finite=True, frozen_parameters_unchanged=True,
            lnc_exact_identity_without_training=True, records=records)

    def test_numerical_failure_retains_exact_partial_state(self):
        original = core.SpectralTriadicGraph.forward
        def fail_third(model, *args):
            if model.triadic_diagnostics['calls'] == 2:
                raise FloatingPointError('Synthetic registered failure')
            return original(model, *args)
        with patch.object(core.SpectralTriadicGraph, 'forward', fail_third):
            with self.assertRaisesRegex(FloatingPointError, 'Synthetic registered failure') as caught:
                experiment.fit(arrays(), small_protocol(), 'full', .001, 112, 'cpu', fixed_epochs=5)
        state, context = caught.exception.training_partial_state, caught.exception.training_failure_context
        self.assertEqual(state['steps'], 2)
        self.assertEqual(context['completed_steps'], 2)
        self.assertEqual(context['current_step'], 3)
        self.assertEqual(context['operation'], 'forward')
        self.assertTrue(all(int(v['step']) == 2 for v in state['optimizer']['state'].values()))
        EVIDENCE['failure'] = dict(exact_completed_steps=2, current_failed_step=3,
            operation='forward', model_optimizer_and_rng_retained=True, formal_fit=False)


if __name__ == '__main__':
    destination = HERE / 'SPECTRAL_TRIADIC_UNIT_V1.json'
    if destination.exists():
        raise ValueError('Preserve existing unit evidence')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SpectralTriadicChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sources = [Path(__file__), Path(core.__file__), Path(experiment.__file__),
        experiment.ROOT / 'test_nested_anchor_v2.py', experiment.ROOT / 'overlap_block_models.py',
        experiment.ROOT / 'sharpness_graph_v1.py', HERE / 'SHONN_AUTHOR_SOURCE_V1.json',
        HERE / 'shonn_author_v1_lib__layers.py']
    record = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        tests=result.testsRun, success=result.wasSuccessful(), failures=len(result.failures),
        errors=len(result.errors), evidence=EVIDENCE,
        source_sha256={str(path): base.digest(path) for path in sources},
        formal_neural_fits=0, evaluation_labels_read=False))
    base.write_json(destination, record)
    raise SystemExit(0 if result.wasSuccessful() else 1)
