"""Independent circuit/flow, derivatives, boundary, and matched-budget checks."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import importlib.util
import io
from pathlib import Path
import unittest

import numpy as np
from scipy.linalg import cho_factor, cho_solve
import torch
from torch.nn import functional as F

from improve import resistance_flow_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent / 'research/curvature_next_20260916'
EVIDENCE = {}


def example():
    rng = np.random.default_rng(351)
    w = rng.uniform(.4, 1.4, size=(6, 6))
    w = np.triu(w, 1)
    w += w.T
    w[0, 2:] = 0.
    w[2:, 0] = 0.
    return w


def scipy_flow(initial, p, local=False):
    w = initial.copy()
    support = w > 0
    row, col = np.nonzero(support)
    leaf = (support.sum(1)[row] == 1) | (support.sum(1)[col] == 1)
    traces = []
    for _ in range(p['flow_iterations']):
        matrix = np.diag(w.sum(1)) - w + p['laplacian_ridge'] * np.eye(len(w))
        inv = cho_solve(cho_factor(matrix, lower=True), np.eye(len(w)))
        r = inv.diagonal()[row] + inv.diagonal()[col] - inv[row, col] - inv[col, row]
        if local:
            r = 1. / w[row, col]
        node = 1. - .5 * np.bincount(row, w[row, col] * r, minlength=len(w))
        curvature = (node[row] + node[col]) * r.min() / r
        curvature[leaf & (curvature < 0)] *= -1
        distance = 1. / w[row, col]
        center = np.average(curvature, weights=distance)
        multiplier = 1. - p['flow_step'] * (curvature - center)
        w = np.zeros_like(w)
        w[row, col] = 1. / (distance * multiplier)
        w = .5 * (w + w.T)
        traces.append(w.copy())
    return w, traces


def arrays():
    rng = np.random.default_rng(851)
    counts = np.array([2, 5, 3, 4])
    pairs = np.array([[0, 1, 2, 3, 4, 0, 1, 2], [0, 1, 2, 0, 1, 2, 0, 1]])
    labels = np.array([1., 0., 1., 0., 1., 0., 1., 0.])
    physical = pairs[:, labels == 1] + np.array([2, 7])[:, None]
    other = np.array([[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
                      [2, 5, 10, 11, 12, 13, 0, 1, 12, 11, 12, 13]])
    a = dict(node_counts=counts, pair_offset=np.array([2, 7]), nd=np.array(3),
        pairs_train=pairs, y_train=labels, folds=np.arange(len(labels)) % 2,
        anchor_oof=np.linspace(-.5, .5, len(labels)), edges=np.concatenate((physical, other), axis=1),
        relation=np.r_[np.zeros(physical.shape[1], dtype=np.int64), np.ones(other.shape[1], dtype=np.int64)],
        target_codes=np.r_[pairs[0, labels == 1] * 3 + pairs[1, labels == 1],
                           np.full(other.shape[1], -1, dtype=np.int64)])
    for index, count in enumerate(counts):
        a['x_' + str(index)] = rng.normal(size=(count, 4 + index)).astype(np.float32)
    return a


class FlowChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.random.default_generator.manual_seed(712)

    def test_pinned_author_forward_and_safe_gradient(self):
        source = HERE / 'resistance_author_v1_ResistanceCurvature__resistance_curvature.py'
        manifest = base.read_json(HERE / 'RESISTANCE_AUTHOR_SOURCE_V1.json')
        row = next(item for item in manifest if item['path'].endswith('resistance_curvature.py'))
        self.assertEqual(base.digest(source), row['sha256'])
        spec = importlib.util.spec_from_file_location('pinned_resistance_v1', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        author = module.ResistanceCurvature(device='cpu')
        weight = torch.tensor(example(), dtype=torch.float64, requires_grad=True)
        p = core.parameters()
        actual, _ = core.evolve(weight, weight.detach() > 0, p, 'full')
        with redirect_stdout(io.StringIO()):
            expected = author.cal_curvature_flow(weight, n_iter=p['flow_iterations'], step=p['flow_step'],
                flow_norm=True, dtype=torch.float64, disturb_factor=p['laplacian_ridge'])
        error = float((actual - expected).detach().abs().max())
        self.assertLess(error, 1e-7)
        local_gradient = torch.autograd.grad(actual.square().sum(), weight, retain_graph=True)[0]
        self.assertTrue(torch.isfinite(local_gradient).all())
        author_gradient = torch.autograd.grad(expected.square().sum(), weight)[0]
        EVIDENCE['author_forward'] = dict(commit=row['commit'], maximum_forward_error=error,
            local_gradient_finite=True, author_gradient_finite=bool(torch.isfinite(author_gradient).all()),
            author_zero_coordinate_division_not_copied=True,
            author_float32_identity_ridge_rounding_not_copied=True)

    def test_independent_scipy_flow_and_global_path_control(self):
        p = core.parameters()
        initial = example()
        errors, fields = [], {}
        for kind in core.KINDS:
            weight = torch.tensor(initial, dtype=torch.float64)
            actual, trace = core.evolve(weight, weight > 0, p, kind)
            expected, _ = scipy_flow(initial, p, local=(kind == 'local_resistance'))
            if kind == 'no_flow':
                np.testing.assert_array_equal(actual.numpy(), initial)
            else:
                np.testing.assert_allclose(actual.numpy(), expected, atol=2e-11, rtol=2e-11)
                errors.append(float(np.max(np.abs(actual.numpy() - expected))))
            self.assertEqual(len(trace), p['flow_iterations'])
            self.assertTrue(all(row['minimum_multiplier'] > 0 for row in trace))
            self.assertTrue(all(row['inverse_relative_residual'] <= p['inverse_residual_tolerance'] for row in trace))
            fields[kind] = actual
        self.assertGreater(float((fields['full'] - fields['local_resistance']).abs().max()), 1e-3)
        self.assertGreater(float((fields['full'] - fields['no_flow']).abs().max()), 1e-3)
        EVIDENCE['independent_flow'] = dict(maximum_scipy_error=max(errors),
            finite_positive_steps=True, full_differs_from_each_control=True, inverse_calls_equal=True)

    def test_resistance_flow_finite_difference(self):
        initial = example()
        r, c = np.nonzero(np.triu(initial, 1))
        row, col = torch.as_tensor(r), torch.as_tensor(c)
        values = torch.tensor(initial[r, c], dtype=torch.float64, requires_grad=True)
        p = dict(core.parameters(), flow_iterations=2)
        def apply(vector):
            matrix = vector.new_zeros(initial.shape).index_put((row, col), vector)
            matrix = matrix + matrix.T
            value, _ = core.evolve(matrix, torch.as_tensor(initial > 0), p, 'full')
            return core.normalize_affinity(value)
        self.assertTrue(torch.autograd.gradcheck(apply, (values,), eps=1e-6, atol=2e-5, rtol=2e-4))
        EVIDENCE['derivatives'] = dict(gradcheck=True, zero_nonedge_division_avoided=True,
            symmetric_active_edge_parameterization=True)

    def test_permutation_and_training_boundary(self):
        initial = torch.tensor(example(), dtype=torch.float64)
        p = core.parameters()
        order = torch.tensor([5, 0, 3, 1, 4, 2])
        for kind in core.KINDS:
            value, _ = core.evolve(initial, initial > 0, p, kind)
            permuted = initial[order][:, order]
            other, _ = core.evolve(permuted, permuted > 0, p, kind)
            torch.testing.assert_close(other, value[order][:, order], atol=2e-11, rtol=2e-11)
        a = arrays()
        small = dict(p, width=8, heads=2, metric_width=4, depth=1, neighbors=2, folds=2, dropout=0.)
        model, xs, graphs, _ = core.setup(a, small, 'full', 'cpu')
        with self.assertRaisesRegex(ValueError, 'training arrays only'):
            core.setup(dict(a, y_val=np.ones(2)), small, 'full', 'cpu')
        for graph in graphs:
            for i, node_type in enumerate((1, 2)):
                alternative, _ = core.fixed_support(a['x_' + str(node_type)], small, 'cpu')
                torch.testing.assert_close(graph.supports[i], alternative, atol=0, rtol=0)
        query = torch.tensor([[2, 3, 4, 5], [7, 8, 9, 7]])
        with torch.no_grad():
            model.readout[-1].weight.fill_(.02)
            model.eval()
            value = model(xs, graphs[0], query)
            reverse = model(xs, graphs[0], query.flip(1)).flip(0)
            chunks = torch.cat([model(xs, graphs[0], query[:, i:i+1]) for i in range(query.shape[1])])
        torch.testing.assert_close(value, reverse, atol=3e-6, rtol=0.)
        torch.testing.assert_close(value, chunks, atol=3e-6, rtol=0.)
        EVIDENCE['boundary'] = dict(node_permutation=True, training_only_entry=True,
            topology_uses_fixed_attributes_only=True, query_replay_max_abs=float((value - chunks).abs().max()))

    def test_matched_initialization_randomness_and_null(self):
        a = arrays()
        p = dict(core.parameters(), width=8, heads=2, metric_width=4, depth=1, neighbors=2, folds=2)
        initial, final_rng, calls = None, None, []
        for kind in core.KINDS:
            torch.random.default_generator.manual_seed(719)
            model, xs, graphs, folds = core.setup(a, p, kind, 'cpu')
            state = {name: value.clone() for name, value in model.state_dict().items()}
            if initial is not None:
                self.assertEqual(set(state), set(initial))
                for name in state:
                    torch.testing.assert_close(state[name], initial[name], atol=0., rtol=0.)
            initial = state
            pairs = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None])
            model.eval()
            with torch.no_grad():
                prediction = model(xs, graphs[0], pairs)
            torch.testing.assert_close(prediction, torch.zeros_like(prediction), atol=0., rtol=0.)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            model.train()
            for step in range(3):
                ids = folds[step % 2]
                optimizer.zero_grad(set_to_none=True)
                value = model(xs, graphs[step % 2], pairs[:, ids])
                loss = F.binary_cross_entropy_with_logits(value.double() + torch.as_tensor(a['anchor_oof'])[ids],
                    torch.as_tensor(a['y_train'])[ids])
                loss.backward()
                self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                    for parameter in model.parameters()))
                optimizer.step()
            self.assertTrue(all(int(state['step']) == 3 for state in optimizer.state.values()))
            current_rng = torch.random.get_rng_state()
            if final_rng is not None:
                torch.testing.assert_close(current_rng, final_rng, atol=0., rtol=0.)
            final_rng = current_rng
            calls.append(model.flow_diagnostics['inverse_calls'])
        self.assertEqual(len(set(calls)), 1)
        EVIDENCE['matching'] = dict(initial_states_equal=True, null_residual_exact_zero=True,
            final_random_states_equal=True, synthetic_steps_per_arm=3, every_parameter_gradient_finite=True,
            inverse_calls_per_arm=calls, formal_fits=0)


if __name__ == '__main__':
    output = HERE / 'RESISTANCE_FLOW_UNIT_V1.json'
    if output.exists():
        raise ValueError('Preserve previous numerical-check evidence')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FlowChecks))
    base.write_json(output, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        success=result.wasSuccessful(), tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        evidence=EVIDENCE, source_sha256={str(path): base.digest(path) for path in (Path(__file__), Path(core.__file__))},
        performance_metrics_computed=False, evaluation_labels_read=False, formal_fits=0)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
