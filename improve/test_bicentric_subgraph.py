"""Query masking, exact local operators, and bicentric mechanism checks."""
import unittest

import numpy as np
import torch

from improve import bicentric_subgraph as method


def toy():
    return np.array([[0, 1, 0, 1, 3], [2, 2, 3, 4, 4]], dtype=np.int64)


class BicentricTests(unittest.TestCase):
    def test_spheres_and_dense_gcn_operator(self):
        context = method.SubgraphContext(toy(), 6)
        nodes, labels, edges, weights = context.extract(0, 1)
        np.testing.assert_array_equal(nodes, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(labels, [0, 1, 3, 2, 4])
        adjacency = context.adjacency[nodes][:, nodes].toarray() + np.eye(len(nodes))
        degree = adjacency.sum(axis=1)
        expected = adjacency / np.sqrt(degree[:, None] * degree[None, :])
        actual = np.zeros_like(expected)
        actual[edges[1], edges[0]] = weights
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        self.assertEqual(adjacency[0, 1], 0)
        with self.assertRaisesRegex(ValueError, 'deleted before'):
            context.extract(0, 2)

    def test_node_permutation_and_query_chunk_invariance(self):
        torch.manual_seed(922)
        model = method.BicentricGraph([4], width=8, depth=2, dropout=0).eval()
        torch.nn.init.normal_(model.output[-1].weight, std=.1)
        x = torch.randn(6, 4)
        pairs = torch.tensor([[0, 2, 3], [1, 4, 2]])
        context = method.SubgraphContext(toy(), 6)
        direct = model([x], context, pairs)
        reverse = model([x], context, pairs.flip(1)).flip(0)
        chunked = torch.cat([model([x], context, pairs[:, i:i + 1]) for i in range(3)])
        torch.testing.assert_close(direct, reverse, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(direct, chunked, atol=1e-6, rtol=1e-6)
        perm = np.array([3, 5, 1, 4, 0, 2])
        xp = torch.empty_like(x)
        xp[torch.as_tensor(perm)] = x
        permuted = method.SubgraphContext(perm[toy()], 6)
        value = model([xp], permuted, torch.as_tensor(perm[pairs.numpy()]))
        torch.testing.assert_close(direct, value, atol=1e-6, rtol=1e-6)
        self.assertGreater(float(direct.abs().max()), 1e-4)

    def test_isolated_endpoints_and_mechanism_gradients(self):
        torch.manual_seed(419)
        context = method.SubgraphContext(toy(), 7)
        x = torch.randn(7, 4)
        pairs = torch.tensor([[0, 5], [1, 6]])
        model = method.BicentricGraph([4], width=8, depth=2, dropout=0)
        torch.nn.init.normal_(model.output[-1].weight, std=.1)
        outputs = {}
        for kind in method.KINDS:
            model.kind = kind
            model.zero_grad(set_to_none=True)
            result = model([x], context, pairs)
            self.assertTrue(torch.isfinite(result).all())
            result.sum().backward()
            self.assertGreater(float(model.project[0].weight.grad.abs().sum()), 0)
            outputs[kind] = result.detach()
            if kind == 'full':
                self.assertGreater(float(model.mark.weight.grad.abs().sum()), 0)
                self.assertTrue(all(float(gate.weight.grad.abs().sum()) > 0 for gate in model.pool))
        for kind in method.KINDS[1:]:
            self.assertFalse(torch.allclose(outputs['full'], outputs[kind]))

    def test_removing_noncenter_topology_changes_messages(self):
        torch.manual_seed(512)
        full = method.SubgraphContext(toy(), 6)
        deleted = method.SubgraphContext(toy()[:, :4], 6)
        a, b = full.extract(0, 1), deleted.extract(0, 1)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        model = method.BicentricGraph([4], width=8, depth=2, dropout=0).eval()
        torch.nn.init.normal_(model.output[-1].weight, std=.1)
        x, pairs = torch.randn(6, 4), torch.tensor([[0], [1]])
        self.assertFalse(torch.allclose(model([x], full, pairs), model([x], deleted, pairs)))
        model.kind = 'no_message'
        torch.testing.assert_close(model([x], full, pairs), model([x], deleted, pairs))


if __name__ == '__main__':
    unittest.main()
