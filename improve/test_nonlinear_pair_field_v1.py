"""Independent dense energy/Hessian and convex-solver checks for pair fields."""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import unittest

import numpy as np
from scipy import optimize, sparse
from scipy.special import expit
import torch

from improve import nonlinear_pair_field_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent / 'research/p_energy_next_20260916'
EVIDENCE = {}


def fixture():
    rng = np.random.default_rng(913)
    matrix = rng.uniform(.1, 1., size=(8, 8))
    matrix = np.triu(matrix * (rng.uniform(size=(8, 8)) < .4), 1)
    matrix = matrix + matrix.T
    matrix[0, 1] = matrix[1, 0] = .6
    graph = core.PairGraph.from_scipy(sparse.csr_matrix(matrix), 'cpu')
    train = np.array([0, 2, 4, 6])
    y, offset = np.array([1., 0., 1., 0.]), np.array([-.7, .4, .8, -.1])
    return matrix, graph, train, y, offset


def independent_problem(matrix, train, y, offset, exponent, p):
    row, col = np.nonzero(np.triu(matrix, 1))
    incidence = np.zeros((len(row), len(matrix)))
    incidence[np.arange(len(row)), row] = 1.
    incidence[np.arange(len(row)), col] = -1.
    weight = matrix[row, col]
    def objective(x):
        g, s = incidence @ x, offset + x[train]
        return np.logaddexp(0., s).sum() - y @ s + p['regularization'] / exponent * (weight @ (abs(g) ** exponent)) + .5 * p['ridge'] * (x @ x)
    def gradient(x):
        g = incidence @ x
        value = p['regularization'] * (incidence.T @ (weight * abs(g) ** (exponent - 2.) * g)) + p['ridge'] * x
        value[train] += expit(offset + x[train]) - y
        return value
    def hessian(x):
        g = incidence @ x
        curvature = p['regularization'] * weight * (exponent - 1.) * abs(g) ** (exponent - 2.)
        h = incidence.T @ (curvature[:, None] * incidence) + p['ridge'] * np.eye(len(x))
        probability = expit(offset + x[train])
        h[train, train] += probability * (1. - probability)
        return h
    return objective, gradient, hessian


class FieldChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.random.default_generator.manual_seed(901)

    def test_energy_gradient_and_sparse_hessian(self):
        matrix, graph, train, y, offset = fixture()
        p = core.parameters()
        value = torch.randn(8, dtype=torch.float64, requires_grad=True)
        ids, labels, anchor = torch.as_tensor(train), torch.as_tensor(y), torch.as_tensor(offset)
        errors = []
        for exponent in (2., 3.):
            loss, actual_gradient = core.value_and_gradient(value, graph, ids, labels, anchor,
                exponent, p['regularization'], p['ridge'])
            gradient = torch.autograd.grad(loss, value, create_graph=True)[0]
            torch.testing.assert_close(actual_gradient, gradient, atol=2e-14, rtol=2e-14)
            objective, dense_gradient, dense_hessian = independent_problem(matrix, train, y, offset, exponent, p)
            self.assertAlmostEqual(float(loss.detach()), objective(value.detach().numpy()), places=12)
            np.testing.assert_allclose(gradient.detach().numpy(), dense_gradient(value.detach().numpy()), atol=2e-14, rtol=2e-14)
            differences = value.detach()[graph.u] - value.detach()[graph.v]
            conductance = p['regularization'] * graph.weight * (exponent - 1.) * differences.abs().pow(exponent - 2.)
            probability = torch.sigmoid(anchor + value.detach()[ids])
            mass = torch.full((8,), p['ridge'], dtype=torch.float64).index_add(0, ids, probability * (1. - probability))
            multiply, diagonal = graph.weighted_operator(conductance, mass)
            direction = torch.randn(8, dtype=torch.float64)
            expected = dense_hessian(value.detach().numpy()) @ direction.numpy()
            np.testing.assert_allclose(multiply(direction).numpy(), expected, atol=2e-14, rtol=2e-14)
            errors.append(float(np.max(np.abs(multiply(direction).numpy() - expected))))
            torch.testing.assert_close(diagonal, torch.from_numpy(np.diag(dense_hessian(value.detach().numpy()))), atol=2e-14, rtol=2e-14)
        EVIDENCE['independent_derivatives'] = dict(exponents=[2., 3.], maximum_hessian_vector_error=max(errors),
            autograd_gradient=True, independent_dense_incidence=True)

    def test_convex_solutions_against_independent_scipy(self):
        matrix, graph, train, y, offset = fixture()
        p = dict(core.parameters(), gradient_tolerance=1e-10)
        start, errors, gaps = None, [], []
        for exponent in (2., 3.):
            field, trace = core.solve(graph, train, y, offset, exponent, p, initial=start)
            objective, gradient, hessian = independent_problem(matrix, train, y, offset, exponent, p)
            result = optimize.minimize(objective, np.zeros(8), jac=gradient, hess=hessian,
                method='trust-exact', options=dict(gtol=1e-11, maxiter=100))
            self.assertLess(np.max(np.abs(gradient(result.x))), 2e-8)
            np.testing.assert_allclose(field.numpy(), result.x, atol=2e-6, rtol=0.)
            self.assertLess(trace[-1]['maximum_gradient'], 1e-10)
            self.assertTrue(all(b['objective'] <= a['objective'] for a, b in zip(trace, trace[1:])))
            errors.append(float(np.max(np.abs(field.numpy() - result.x))))
            gaps.append(abs(objective(field.numpy()) - result.fun))
            if start is not None:
                self.assertGreater(float((field - start).abs().max()), .01)
            start = field
        EVIDENCE['independent_solutions'] = dict(maximum_solution_error=max(errors), maximum_objective_gap=max(gaps),
            scipy_trust_exact_reference=True, monotone_armijo=True, nonlinear_differs_from_quadratic=True)

    def test_neighbors_keep_ties_and_do_not_read_labels(self):
        x = np.array([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        graph, audit = core.entity_neighbors(x, k=1)
        self.assertEqual(graph[0].nnz, 2)
        self.assertGreater(audit['ties_kept'], 0)
        order = np.array([4, 2, 0, 3, 1])
        other, _ = core.entity_neighbors(x[order], k=1)
        np.testing.assert_allclose(other.toarray(), graph.toarray()[np.ix_(order, order)], atol=1e-14, rtol=0.)
        rng = np.random.default_rng(928)
        a = dict(node_counts=np.array([2, 4, 3, 2]), pair_offset=np.array([2, 6]), nd=np.array(3),
            x_1=rng.normal(size=(4, 5)), x_2=rng.normal(size=(3, 7)),
            y_train=np.array([1., 0., 1.]), pairs_train=np.array([[0, 1, 2], [0, 1, 2]]),
            edges=np.array([[2, 3], [6, 7]]), relation=np.array([3, 3]), target_codes=np.array([0, 4]))
        first, _ = core.training_graph(a)
        second, _ = core.training_graph(dict(a, y_train=1. - a['y_train'], edges=a['edges'][:, :0],
            relation=a['relation'][:0], target_codes=a['target_codes'][:0]))
        np.testing.assert_array_equal(first.toarray(), second.toarray())
        with self.assertRaisesRegex(ValueError, 'training arrays only'):
            core.training_graph(dict(a, y_val=np.ones(1)))
        EVIDENCE['graph_boundary'] = dict(all_distance_ties_retained=True, relabeling_equivariant=True,
            every_target_label_and_edge_irrelevant_to_graph=True, training_only_entry=True)

    def test_orientation_permutation_and_query_replay(self):
        matrix, graph, train, y, offset = fixture()
        p = dict(core.parameters(), gradient_tolerance=1e-10)
        first, _ = core.solve(graph, train, y, offset, 3., p)
        swapped, _ = core.solve(replace(graph, u=graph.v, v=graph.u), train, y, offset, 3., p)
        torch.testing.assert_close(first, swapped, atol=1e-8, rtol=0.)
        order = np.array([7, 3, 1, 6, 2, 0, 4, 5])
        inverse = np.argsort(order)
        permuted = core.PairGraph.from_scipy(sparse.csr_matrix(matrix[np.ix_(order, order)]), 'cpu')
        second, _ = core.solve(permuted, inverse[train], y, offset, 3., p)
        np.testing.assert_allclose(second.numpy()[inverse], first.numpy(), atol=2e-8, rtol=0.)
        pairs = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
        prediction = core.predict(first, pairs, 2)
        torch.testing.assert_close(core.predict(first, pairs.flip(1), 2).flip(0), prediction, atol=0., rtol=0.)
        torch.testing.assert_close(torch.cat([core.predict(first, pairs[:, i:i+1], 2) for i in range(4)]), prediction, atol=0., rtol=0.)
        EVIDENCE['equivariance_and_replay'] = dict(edge_orientation=True, node_permutation=True,
            reordered_and_batched_lookup_bitwise_equal=True)

    def test_isolated_nodes_are_coercive_and_unlabeled_zero(self):
        graph = core.PairGraph.from_scipy(sparse.csr_matrix((4, 4), dtype=np.float64), 'cpu')
        p = dict(core.parameters(), gradient_tolerance=1e-10)
        for exponent in (2., 3.):
            field, trace = core.solve(graph, [0], [1.], [-.7], exponent, p)
            expected = optimize.brentq(lambda value: expit(-.7 + value) - 1. + p['ridge'] * value, 0., 20.)
            self.assertAlmostEqual(float(field[0]), expected, places=7)
            torch.testing.assert_close(field[1:], torch.zeros(3, dtype=torch.float64), atol=0., rtol=0.)
            self.assertLessEqual(trace[-1]['maximum_gradient'], p['gradient_tolerance'])
        EVIDENCE['empty_support_graph'] = dict(independent_scalar_root=True, unlabeled_field_exactly_zero=True,
            strictly_positive_all_node_ridge=True)


if __name__ == '__main__':
    output = HERE / 'NONLINEAR_FIELD_UNIT_V1.json'
    if output.exists():
        raise ValueError('Preserve previous nonlinear-field evidence')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FieldChecks))
    sources = [Path(__file__), Path(core.__file__)]
    base.write_json(output, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), success=result.wasSuccessful(),
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), evidence=EVIDENCE,
        source_sha256={str(path): base.digest(path) for path in sources},
        performance_metrics_computed=False, evaluation_labels_read=False, formal_fits=0)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
