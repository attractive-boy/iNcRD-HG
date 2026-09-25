"""Numerical and isolation checks for the conditional meta-learning mechanism."""
import copy
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from improve import episodic_kernel_graph_v1 as core


def fixture():
    rng = np.random.default_rng(31)
    pairs = np.array([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 1, 2, 2, 0, 0, 2, 1, 2]])
    y = np.array([0, 1] * 5, dtype=np.float32)
    positive = pairs[:, y == 1]
    counts = np.array([2, 5, 3, 2])
    offset = np.array([2, 7])
    target = positive + offset[:, None]
    side = np.array([[0, 1, 10], [2, 4, 11]])
    a = dict(pairs_train=pairs, y_train=y, folds=np.repeat(np.arange(5), 2),
        node_counts=counts, pair_offset=offset, nd=np.array(3),
        edges=np.concatenate((target, side), axis=1),
        relation=np.array([0] * len(positive[0]) + [1] * len(side[0])),
        target_codes=np.r_[positive[0] * 3 + positive[1], [-1] * len(side[0])])
    a.update({'x_' + str(i): rng.normal(size=(n, 4)).astype(np.float32) for i, n in enumerate(counts)})
    return a


class EpisodicChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(77)

    def test_primal_matches_dual_predictions_and_gradients(self):
        s = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
        q = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        y = torch.tensor([0, 1, 1, 0, 1, 0, 1])
        actual = core.ridge_scores(s, y, q, .7)
        x = torch.cat((s, torch.ones(7, 1, dtype=s.dtype)), 1)
        z = torch.cat((q, torch.ones(3, 1, dtype=s.dtype)), 1)
        expected = z @ x.T @ torch.linalg.solve(x @ x.T + .7 * torch.eye(7, dtype=s.dtype),
                                               F.one_hot(y, 2).double())
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        ga = torch.autograd.grad(actual.square().sum(), (s, q), retain_graph=True)
        ge = torch.autograd.grad(expected.square().sum(), (s, q))
        for a, b in zip(ga, ge):
            torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
        self.assertTrue(torch.autograd.gradcheck(lambda s, q: core.ridge_scores(s, y, q, .7), (s, q)))

    def test_control_preserves_forward_but_removes_only_support_gradient(self):
        s = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
        q = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        y = torch.tensor([0, 1] * 4)
        full = core.ridge_scores(s, y, q, 1.)
        control = core.ridge_scores(s, y, q, 1., True)
        self.assertTrue(torch.equal(full, control))
        fs, fq = torch.autograd.grad(full.square().sum(), (s, q), retain_graph=True)
        cs, cq = torch.autograd.grad(control.square().sum(), (s, q), allow_unused=True)
        self.assertIsNone(cs)
        self.assertGreater(float(fs.abs().sum()), 0.)
        torch.testing.assert_close(fq, cq, atol=0, rtol=0)

    def test_all_episodes_delete_conditioning_and_query_facts(self):
        a = fixture()
        p = dict(core.parameters(), width=8, heads=2, depth=1)
        model, xs, episodes, folds = core.setup(a, p, 'full', 'cpu')
        self.assertEqual(len(episodes), 20)
        for episode, queries in zip(episodes, folds):
            self.assertFalse(np.intersect1d(episode['support_ids'], queries.numpy()).size)
            deleted = a['pairs_train'][:, np.r_[episode['support_ids'], queries.numpy()]] + a['pair_offset'][:, None]
            arcs = set(map(tuple, episode['graph'][0].numpy().T))
            for r, d in deleted.T:
                self.assertNotIn((r, d), arcs)
                self.assertNotIn((d, r), arcs)
            altered = {k: v.copy() for k, v in a.items()}
            altered['y_train'][queries.numpy()] = 1 - altered['y_train'][queries.numpy()]
            _, _, other, _ = core.setup(altered, p, 'full', 'cpu')
            match = other[episode['query_fold'] * 4 +
                [s for s in range(5) if s != episode['query_fold']].index(episode['support_fold'])]
            self.assertTrue(torch.equal(episode['support_y'], match['support_y']))
            for x, y in zip(episode['graph'], match['graph']):
                self.assertTrue(torch.equal(x, y))

    def test_full_model_replay_query_order_and_identity(self):
        a = fixture()
        p = dict(core.parameters(), width=8, heads=2, depth=1)
        model, xs, episodes, _ = core.setup(a, p, 'full', 'cpu')
        query = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None])
        model.eval()
        z = model(xs, episodes[0], query)
        flipped = model(xs, episodes[0], query.flip(1)).flip(0)
        chunked = torch.cat([model(xs, episodes[0], query[:, :4]), model(xs, episodes[0], query[:, 4:])])
        torch.testing.assert_close(z, flipped, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(z, chunked, atol=1e-6, rtol=1e-6)
        matched = copy.deepcopy(model)
        matched.kind = 'stop_support_gradient'
        torch.testing.assert_close(z, matched(xs, episodes[0], query), atol=0, rtol=0)
        a['pair_offset'] = np.array([0, 7])
        identity, inputs, graphs, _ = core.setup(a, p, 'full', 'cpu')
        self.assertEqual(list(identity.parameters()), [])
        self.assertEqual(inputs, [])
        self.assertTrue(torch.equal(identity(inputs, graphs[0], query), torch.zeros(query.shape[1])))


if __name__ == '__main__':
    unittest.main()
