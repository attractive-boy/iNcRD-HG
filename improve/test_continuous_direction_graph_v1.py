"""Author-operator parity, independent derivatives, direction and label boundaries."""
import ast
from pathlib import Path
from typing import Optional
import unittest

import numpy as np
import torch

from improve import continuous_direction_graph_v1 as core
from improve import anchor_direction_experiment_v1 as experiment
from improve import tabular_foundation as base


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


def dense_operator(theta, edges, n, kind):
    source, target, incoming, outgoing = core.normalized_directions(theta, edges, n, kind)
    left = theta.new_zeros((n, n)).index_put((target, source), incoming, accumulate=True)
    right = theta.new_zeros((n, n)).index_put((target, source), outgoing, accumulate=True)
    return left, right


class DirectionChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(224)
        self.edges = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]])
        self.theta = torch.tensor([.13, .57, 1.12, 1.4], dtype=torch.float64, requires_grad=True)
        self.value = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)

    def test_pinned_author_operator_and_gradients(self):
        path = Path(__file__).resolve().parent / 'research/directional_next_20260916/coed_author_v1_src__utils__fuzzy_laplacian.py'
        self.assertEqual(base.digest(path), 'ff3471823134f0915095d89bfd36c64db00344cd124a6a09276bc36be31ee948')
        parsed = ast.parse(path.read_text())
        function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == 'get_fuzzy_laplacian')
        namespace = dict(torch=torch, Optional=Optional)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        author_edges, author_weights = namespace['get_fuzzy_laplacian'](
            self.edges, self.theta, 5, len(self.theta), torch.ones_like(self.theta), False)
        source, target, incoming, outgoing = core.normalized_directions(self.theta, self.edges, 5, 'full')
        torch.testing.assert_close(torch.stack((source, target)), author_edges, atol=0, rtol=0)
        for actual, expected in zip((incoming, outgoing), author_weights):
            torch.testing.assert_close(actual, expected[:, 0], atol=1e-14, rtol=1e-14)
        upstream = torch.randn_like(self.value)
        actual = core.aggregate(self.value, source, target, incoming) + .37 * core.aggregate(self.value, source, target, outgoing)
        expected = core.aggregate(self.value, *author_edges, author_weights[0][:, 0]) + .37 * core.aggregate(self.value, *author_edges, author_weights[1][:, 0])
        got = torch.autograd.grad((actual * upstream).sum(), (self.theta, self.value), retain_graph=True)
        want = torch.autograd.grad((expected * upstream).sum(), (self.theta, self.value))
        for left, right in zip(got, want):
            torch.testing.assert_close(left, right, atol=2e-13, rtol=2e-13)
        self.assertEqual(float(actual[4].abs().sum()), 0.)

    def test_finite_differences_transpose_and_storage_reversal(self):
        left, right = dense_operator(self.theta, self.edges, 5, 'full')
        torch.testing.assert_close(left, right.T, atol=1e-14, rtol=1e-14)
        flipped, flipped_reverse = dense_operator(np.pi / 2 - self.theta, self.edges.flip(0), 5, 'full')
        torch.testing.assert_close(left, flipped, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(right, flipped_reverse, atol=1e-14, rtol=1e-14)
        # Form unnormalized directed masses independently, then normalize densely.
        matrix = self.theta.new_zeros((5, 5))
        u, v = self.edges
        matrix = matrix.index_put((v, u), self.theta.cos().square())
        matrix = matrix.index_put((u, v), self.theta.sin().square())
        indegree = matrix.sum(1) + 1e-12
        outdegree = matrix.sum(0) + 1e-12
        si = torch.where(indegree < 1e-11, 0., indegree.rsqrt())
        so = torch.where(outdegree < 1e-11, 0., outdegree.rsqrt())
        torch.testing.assert_close(left, si[:, None] * matrix * so[None, :], atol=1e-14, rtol=1e-14)
        self.assertTrue(torch.autograd.gradcheck(lambda t, x: dense_operator(t, self.edges, 5, 'full')[0] @ x,
            (self.theta, self.value), eps=1e-5, atol=2e-6, rtol=2e-5))
        symmetric, symmetric_reverse = dense_operator(self.theta, self.edges, 5, 'symmetric_weights')
        torch.testing.assert_close(symmetric, symmetric.T, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(symmetric, symmetric_reverse, atol=0, rtol=0)
        self.assertGreater(float((left - symmetric).abs().max()), .05)

    def test_same_initialization_and_nonzero_head_query_invariance(self):
        a, p = example(), dict(core.parameters(), width=8, depth=2, dropout=0.)
        models = []
        for kind in core.KINDS:
            torch.manual_seed(931)
            model, xs, graphs, _ = core.setup(a, p, kind, 'cpu')
            models.append(model)
        for model in models[1:]:
            for name, value in models[0].state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
        pairs = torch.tensor([[2, 3, 4, 2], [5, 6, 5, 6]])
        with torch.no_grad():
            models[0].readout[-1].weight.normal_(std=.1)
            for model in models[1:]:
                model.load_state_dict(models[0].state_dict())
            for model in models:
                model.eval()
            expected = models[0](xs, graphs[0], pairs)
            for model in models:
                torch.testing.assert_close(model(xs, graphs[0], pairs), expected, atol=1e-7, rtol=1e-7)
            model = models[0]
            model.theta.copy_(torch.linspace(.15, 1.35, len(model.theta)))
            normal = model(xs, graphs[0], pairs)
            reverse = model(xs, graphs[0], pairs.flip(1)).flip(0)
            chunks = torch.cat([model(xs, graphs[0], pairs[:, i:i + 1]) for i in range(pairs.shape[1])])
            torch.testing.assert_close(normal, reverse, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(normal, chunks, atol=1e-6, rtol=1e-6)

    def test_query_fact_absence_and_inactive_angle_gradient(self):
        a, p = example(), dict(core.parameters(), width=8, depth=2, dropout=0.)
        model, xs, graphs, folds = core.setup(a, p, 'full', 'cpu')
        graph = graphs[0]
        self.assertNotIn(0, graph.edge_ids.tolist())
        changed = {k:v.copy() for k,v in a.items()}
        changed['y_train'][a['folds'] == 0] = 1. - changed['y_train'][a['folds'] == 0]
        other = core.context(changed, folds[0].numpy(), 'cpu')
        for field in ('edges', 'edge_ids', 'relation', 'counts'):
            torch.testing.assert_close(getattr(graph, field), getattr(other, field), atol=0, rtol=0)
        model.readout[-1].weight.data.normal_(std=.2)
        model.eval()
        pairs = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None])
        model(xs, graph, pairs).square().sum().backward()
        self.assertEqual(float(model.theta.grad[0]), 0.)
        self.assertGreater(float(model.theta.grad[graph.edge_ids].abs().sum()), 0.)
        a['pair_offset'] = np.array([0, 5])
        identity, ix, ig, _ = core.setup(a, p, 'full', 'cpu')
        self.assertFalse(list(identity.parameters()))
        self.assertEqual(ix, [])
        self.assertTrue(all(item is None for item in ig))
        torch.testing.assert_close(identity([], None, pairs), torch.zeros(pairs.shape[1]), atol=0, rtol=0)

    def test_training_projection_optimizer_and_training_only_interface(self):
        a = example()
        a['anchor_oof'] = np.linspace(-1., 1., len(a['y_train']))
        p = dict(smoke=True, family='continuous_direction', implementation_module='continuous_direction_graph_v1',
            source_sha256={'continuous_direction_graph_v1.py': base.digest(core.__file__)},
            core_parameters=dict(core.parameters(), width=8, depth=2, dropout=0.), kinds=list(core.KINDS),
            epochs=6, check_interval=3, weight_decay=.001, strong_weight=.1)
        with self.assertRaisesRegex(ValueError, 'training inputs only'):
            experiment.fit(dict(a, y_val=np.zeros(2)), p, 'full', .003, 83, 'cpu', fixed_epochs=6)
        for kind in core.KINDS:
            model, value, trace, detail = experiment.fit(a, p, kind, .003, 83, 'cpu', fixed_epochs=6)
            self.assertIsNone(value)
            self.assertEqual(detail['training_steps'], 6)
            self.assertEqual(detail['direction_gradient_steps'], 0 if kind == 'fixed_direction' else 6)
            self.assertEqual(detail['direction_projection_steps'], 6)
            self.assertTrue(all('anchor_composed' not in row for row in trace))
            optimizer = experiment.optimizer_for(model, p, .003)
            self.assertIs(optimizer.param_groups[1]['params'][0], model.theta)
            self.assertEqual(optimizer.param_groups[1]['lr'], .01)
            self.assertEqual(optimizer.param_groups[1]['weight_decay'], 0.)
            if kind == 'fixed_direction':
                self.assertEqual(detail['direction_summary']['mean_abs_angle_change'], 0.)
            else:
                self.assertGreater(detail['direction_summary']['mean_abs_angle_change'], 0.)
            self.assertEqual(model._training_final_state['steps'], 6)
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, model._training_final_state['model'][name], atol=0, rtol=0)
            with torch.no_grad():
                model.theta[0], model.theta[1] = -2., 3.
            model.project_directions()
            self.assertEqual(float(model.theta[0]), 0.)
            self.assertAlmostEqual(float(model.theta[1]), np.pi / 2, places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
