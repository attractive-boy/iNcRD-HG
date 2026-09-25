"""Independent fixed-point, derivative, receptive-field and isolation checks."""
import unittest

import numpy as np
import scipy.sparse as sp
import torch

from improve import equilibrium_graph_v1 as core


def dense_solution(rhs, factor, transition, gamma):
    coupling = core.normalized_gram(factor)
    matrix = torch.eye(rhs.numel(), dtype=rhs.dtype) - gamma * torch.kron(transition.contiguous(), coupling.T.contiguous())
    return torch.linalg.solve(matrix, rhs.flatten()).reshape_as(rhs)


def example():
    rng = np.random.default_rng(98)
    a = dict(node_counts=np.array([2, 3, 2, 1]), pair_offset=np.array([2, 5]), nd=np.array(2),
        pairs_train=np.array([[0, 1, 2, 0, 1, 2], [0, 0, 1, 1, 1, 0]]),
        y_train=np.array([1., 0., 1., 0., 1., 0.], dtype=np.float32),
        folds=np.array([0, 1, 2, 3, 4, 0]),
        edges=np.array([[2, 4, 3, 0, 1, 2, 3, 0], [5, 6, 6, 2, 3, 7, 4, 5]]),
        relation=np.array([0, 0, 0, 1, 1, 2, 2, 1]),
        target_codes=np.array([0, 5, 3, -1, -1, -1, -1, -1]))
    for i, n in enumerate(a['node_counts']):
        a['x_' + str(i)] = rng.normal(size=(n, 4 + i)).astype(np.float32)
    return a


class EquilibriumChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(131)
        self.p = dict(core.parameters(), solver_tolerance=1e-12, width=8)
        self.transition = torch.tensor([[.5, .5, 0., 0.], [.2, .3, .5, 0.],
            [.0, .4, .6, 0.], [0., 0., 0., 1.]], dtype=torch.float64)
        self.adj = self.transition.to_sparse_csr()
        self.transposed = self.transition.T.to_sparse_csr()

    def test_fixed_point_matches_dense_kronecker_and_is_contractively_bounded(self):
        rhs = torch.randn(4, 3, dtype=torch.float64)
        factor = torch.randn(3, 3, dtype=torch.float64)
        coupling = core.normalized_gram(factor)
        spectrum = torch.linalg.eigvalsh(coupling)
        self.assertGreaterEqual(float(spectrum.min()), -1e-12)
        self.assertLessEqual(float(spectrum.max()), 1.)
        record = {}
        actual = core.equilibrium(rhs, factor, self.adj, self.transposed, self.p, 'full', record)
        expected = dense_solution(rhs, factor, self.transition, self.p['gamma'])
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        self.assertLessEqual(record['forward']['relative_residual'], self.p['solver_tolerance'])

    def test_implicit_derivatives_match_dense_solve_and_finite_differences(self):
        rhs = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
        factor = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        upstream = torch.randn_like(rhs)
        record = {}
        actual = core.equilibrium(rhs, factor, self.adj, self.transposed, self.p, 'full', record)
        got = torch.autograd.grad((actual * upstream).sum(), (rhs, factor))
        expected = dense_solution(rhs, factor, self.transition, self.p['gamma'])
        want = torch.autograd.grad((expected * upstream).sum(), (rhs, factor))
        for left, right in zip(got, want):
            torch.testing.assert_close(left, right, atol=1e-9, rtol=1e-9)
        self.assertLessEqual(record['backward']['relative_residual'], self.p['solver_tolerance'])
        self.assertTrue(torch.autograd.gradcheck(lambda b, f: core.equilibrium(
            b, f, self.adj, self.transposed, self.p, 'full', {}), (rhs, factor),
            eps=1e-5, atol=2e-6, rtol=2e-5))

    def test_two_hop_control_and_long_range_signal(self):
        edges = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
        channels, _ = core.relation_channels(edges, np.zeros(4, dtype=np.int64), 5, 1, 'cpu')
        adj, transposed = channels[0]
        rhs = torch.zeros(5, 1, dtype=torch.float64)
        rhs[0] = 1.
        factor = torch.ones(1, 1, dtype=torch.float64)
        control = core.equilibrium(rhs, factor, adj, transposed, self.p, 'two_hops', {})
        g = core.normalized_gram(factor)
        first = self.p['gamma'] * (adj @ rhs) @ g
        expected = rhs + first + self.p['gamma'] * (adj @ first) @ g
        torch.testing.assert_close(control, expected, atol=1e-12, rtol=1e-12)
        full = core.equilibrium(rhs, factor, adj, transposed, self.p, 'full', {})
        self.assertEqual(float(control[4]), 0.)
        self.assertGreater(float(full[4]), .005)

    def test_solver_rejects_unconverged_or_invalid_configuration(self):
        rhs = torch.ones(4, 2, dtype=torch.float64)
        factor = torch.eye(2, dtype=torch.float64)
        with self.assertRaises(ArithmeticError):
            core.equilibrium(rhs, factor, self.adj, self.transposed,
                             dict(self.p, solver_max_iterations=1), 'full', {})
        with self.assertRaises(ValueError):
            core.equilibrium(rhs, factor, self.adj, self.transposed,
                             dict(self.p, gamma=1.), 'full', {})

    def test_masking_matched_initialization_and_query_invariance(self):
        a = example()
        torch.manual_seed(752)
        model, xs, graphs, folds = core.setup(a, self.p, 'full', 'cpu')
        torch.manual_seed(752)
        control, _, _, _ = core.setup(a, self.p, 'two_hops', 'cpu')
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, control.state_dict()[key], atol=0, rtol=0)
        self.assertEqual(float(graphs[0][0][0][0].to_dense()[2, 5]), 0.)
        changed = {k: v.copy() for k, v in a.items()}
        changed['y_train'][a['folds'] == 0] = 1. - changed['y_train'][a['folds'] == 0]
        keep = changed['target_codes'] != 0
        changed['edges'] = changed['edges'][:, keep]
        changed['relation'], changed['target_codes'] = changed['relation'][keep], changed['target_codes'][keep]
        _, _, new_graphs, _ = core.setup(changed, self.p, 'full', 'cpu')
        for (adj, transpose), (other, other_t) in zip(graphs[0][0], new_graphs[0][0]):
            torch.testing.assert_close(adj.to_dense(), other.to_dense(), atol=0, rtol=0)
            torch.testing.assert_close(transpose.to_dense(), other_t.to_dense(), atol=0, rtol=0)
        torch.testing.assert_close(graphs[0][1], new_graphs[0][1], atol=0, rtol=0)
        model.eval()
        with torch.no_grad():
            model.readout[-1].weight.normal_(std=.1)
            pairs = torch.tensor([[2, 3, 4], [5, 6, 5]])
            normal = model(xs, graphs[0], pairs)
            reverse = model(xs, graphs[0], pairs.flip(1)).flip(0)
            chunks = torch.cat([model(xs, graphs[0], pairs[:, i:i + 1]) for i in range(3)])
            torch.testing.assert_close(normal, reverse, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(normal, chunks, atol=1e-6, rtol=1e-6)
        a['pair_offset'] = np.array([0, 5])
        identity, identity_x, identity_g, _ = core.setup(a, self.p, 'full', 'cpu')
        self.assertEqual(sum(v.numel() for v in identity.parameters()), 0)
        self.assertEqual(identity_x, [])
        torch.testing.assert_close(identity([], None, pairs), torch.zeros(3), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
