import unittest

import numpy as np
import torch

from improve import masked_relation_graph as model


def toy():
    return dict(node_counts=np.array([6, 5]), nd=np.array(5), pair_offset=np.array([0, 6]),
        pairs_train=np.array([[0, 1, 2, 3], [0, 1, 2, 3]]), y_train=np.array([1, 1, 1, 0]),
        folds=np.array([0, 1, 1, 0]),
        edges=np.array([[0, 1, 2, 0, 1, 2, 3, 4], [6, 7, 8, 1, 2, 3, 4, 5]]),
        relation=np.array([0, 0, 0, 1, 1, 1, 1, 1]), target_codes=np.array([0, 6, 12, -1, -1, -1, -1, -1]),
        x_0=np.ones((6, 3), dtype=np.float32), x_1=np.ones((5, 4), dtype=np.float32))


class RelationMaskTests(unittest.TestCase):
    def test_queries_and_auxiliary_edges_are_absent_before_counts(self):
        a = toy()
        graph = model.context(a, 0, 'full', 'cpu', .6)
        edges, _, counts = graph[:3]
        codes = set((edges[0] * 11 + edges[1]).tolist())
        self.assertNotIn(6, codes)
        self.assertNotIn(66, codes)
        self.assertEqual(counts[0, 0].item(), 0.)
        for positive, negative, rel in graph[3]:
            self.assertEqual(rel, 1)
            for r, d in positive.T.tolist():
                self.assertNotIn(r * 11 + d, codes)
                self.assertNotIn(d * 11 + r, codes)
            for r, d in negative.T.tolist():
                self.assertLess(r, 6)
                self.assertLess(d, 6)
                self.assertNotEqual(r, d)
                self.assertFalse(np.any((a['edges'][0] == r) & (a['edges'][1] == d)))

    def test_shuffled_null_preserves_graph_and_marginal_targets(self):
        a = toy()
        full = model.context(a, 0, 'full', 'cpu', .6)
        null = model.context(a, 0, 'shuffled_aux', 'cpu', .6)
        for x, y in zip(full[:3], null[:3]):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
        torch.testing.assert_close(full[4], null[4], atol=0, rtol=0)
        for (p, n, r), (q, z, s) in zip(full[3], null[3]):
            self.assertEqual(r, s)
            torch.testing.assert_close(torch.sort(p[0]).values, torch.sort(q[0]).values)
            torch.testing.assert_close(torch.sort(p[1]).values, torch.sort(q[1]).values)
            torch.testing.assert_close(n, z)

    def test_full_auxiliary_reaches_encoder_bce_does_not(self):
        a = toy()
        p = dict(folds=2, width=8, depth=1, dropout=0., mask_fraction=.6)
        for kind in ('full', 'bce'):
            torch.manual_seed(3)
            network, xs, graphs, _ = model.setup(a, p, kind, 'cpu')
            network.train()
            network(xs, graphs[0], torch.tensor([[0], [6]]))
            loss, _ = network.auxiliary(xs, graphs[0], torch.Generator().manual_seed(4))
            loss.backward()
            total = sum(v.grad.abs().sum().item() for v in network.project.parameters() if v.grad is not None)
            self.assertGreater(total, 0.) if kind == 'full' else self.assertEqual(total, 0.)
            self.assertIsNone(network._cached_h)


if __name__ == '__main__':
    unittest.main()
