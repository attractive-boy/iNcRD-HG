"""Exact simple-path enumeration, mechanism separation, and query isolation."""
from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np
import torch

from improve import loopy_path_graph_v1 as loopy
from improve.test_nested_anchor_v2 import fixture


class LoopyPathTests(unittest.TestCase):
    def test_complete_enumeration_matches_independent_neighbor_walk(self):
        pairs = np.indices((4, 3)).reshape(2, -1)
        pairs = pairs[:, ~((pairs[0] == 0) & (pairs[1] == 1))]
        offsets = np.array([2, 9])
        centers, paths = loopy.rooted_paths(pairs, 4, 3, offsets)
        edges = pairs + offsets[:, None]
        adjacent = {i: set() for i in range(12)}
        for u, v in edges.T:
            adjacent[u].add(v); adjacent[v].add(u)
        brute = set()
        for center, neighbors in adjacent.items():
            for first in neighbors:
                for middle in adjacent[first] - {center}:
                    for last in adjacent[middle] & neighbors - {first}:
                        brute.add((center, first, middle, last))
        actual = [tuple(row) for row in np.column_stack((centers, paths))]
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(set(actual), brute)
        self.assertTrue(all(len(set(row)) == 4 for row in actual))
        self.assertTrue(all((c, p[2], p[1], p[0]) in brute for c, p in zip(centers, paths)))
        b = np.zeros((4, 3), dtype=np.int64); b[pairs[0], pairs[1]] = 1
        shared = (b.T @ b)[np.triu_indices(3, 1)]
        self.assertEqual(len(actual), int(8 * (shared * (shared-1) // 2).sum()))

    def test_path_message_changes_joint_assignments_with_identical_node_marginals(self):
        torch.manual_seed(281)
        full = loopy.PathGIN(8, 'full').double()
        control = deepcopy(full); control.kind = 'no_path_message'
        self.assertEqual(set(full.state_dict()), set(control.state_dict()))
        h = torch.randn(6, 8, dtype=torch.float64, requires_grad=True)
        first = h[torch.tensor([[0, 4, 1], [2, 5, 3]])]
        second = h[torch.tensor([[0, 5, 1], [2, 4, 3]])]
        torch.testing.assert_close(control(first).sum(0), control(second).sum(0), atol=1e-12, rtol=1e-12)
        self.assertGreater((full(first).sum(0)-full(second).sum(0)).abs().max().item(), 1e-4)
        torch.testing.assert_close(full(first), full(first.flip(1)), atol=1e-12, rtol=1e-12)
        for model in (full, control):
            output = model(first)
            gradients = torch.autograd.grad(output.square().sum(), tuple(model.parameters()), retain_graph=True)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(gradients[0].abs().sum().item(), 0.)
            self.assertGreater(model.transform.weight.numel(), 0)

    def test_held_out_facts_and_labels_are_absent_from_physical_and_cycle_support(self):
        a, _ = fixture()
        omitted = np.flatnonzero(a['folds'] == 0)
        graph = loopy.context(a, omitted, 'cpu')
        changed = {k: v.copy() for k, v in a.items()}
        changed['y_train'][omitted] = 1 - changed['y_train'][omitted]
        changed['y_val'] = np.array([101, 999])
        changed['anchor_oof'] = np.full(len(a['y_train']), 333.)
        alternative = loopy.context(changed, omitted, 'cpu')
        for name in ('edges', 'relation', 'counts', 'centers', 'paths', 'path_degree'):
            torch.testing.assert_close(getattr(graph, name), getattr(alternative, name), atol=0, rtol=0)
        forbidden = set(map(tuple, (a['pairs_train'][:, omitted] + a['pair_offset'][:, None]).T))
        forbidden |= {e[::-1] for e in list(forbidden)}
        physical = set(map(tuple, graph.edges.numpy().T))
        self.assertFalse(physical & forbidden)
        for c, p in zip(graph.centers.tolist(), graph.paths.tolist()):
            self.assertNotIn(c, p)
            cycle = [c] + p + [c]
            self.assertFalse(set(zip(cycle[:-1], cycle[1:])) & forbidden)
        self.assertFalse(np.isin(graph.support_positive_indices, omitted).any())

    def test_zero_null_replay_gradients_and_reindexing(self):
        torch.manual_seed(539)
        a, _ = fixture()
        p = loopy.parameters(); p.update(width=8, heads=2, depth=2, path_chunk=9, dropout=0.)
        model, xs, graphs, folds = loopy.setup(a, p, 'full', 'cpu')
        query = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None])[:, folds[0]]
        model.eval()
        null = model(xs, graphs[0], query)
        torch.testing.assert_close(null, torch.zeros_like(null), atol=0, rtol=0)
        torch.nn.init.normal_(model.readout[-1].weight, std=.2)
        expected = model(xs, graphs[0], query)
        expected.square().sum().backward()
        self.assertGreater(model.layers[0].path.transform.weight.grad.abs().sum().item(), 0.)
        reversed_value = model(xs, graphs[0], query.flip(1)).flip(0)
        chunks = torch.cat([model(xs, graphs[0], query[:, i:i+3]) for i in range(0, query.shape[1], 3)])
        torch.testing.assert_close(expected, reversed_value, atol=3e-7, rtol=3e-7)
        torch.testing.assert_close(expected, chunks, atol=3e-7, rtol=3e-7)
        orders = [torch.randperm(len(x)) for x in xs]
        offsets = np.r_[0, np.cumsum(a['node_counts'])[:-1]]
        permutation = torch.cat([order + int(offset) for order, offset in zip(orders, offsets)])
        inverse = permutation.argsort()
        graph = graphs[0]
        moved = replace(graph, edges=inverse[graph.edges], counts=graph.counts[permutation],
            centers=inverse[graph.centers], paths=inverse[graph.paths], path_degree=graph.path_degree[permutation])
        actual = model([x[order] for x, order in zip(xs, orders)], moved, inverse[query])
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-7)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
