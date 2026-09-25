import unittest

import numpy as np
import torch

from improve.equivariant_hypergraph import incidence_arrays, member_messages, segment_sum, context, setup


class EquivariantHypergraphTests(unittest.TestCase):
    def test_relation_stars_are_sets_and_pairwise_is_a_different_diagnostic(self):
        edges = np.array([[0, 0, 0, 2], [1, 1, 2, 3]])
        relations = np.array([0, 0, 0, 1])
        stars = incidence_arrays(edges, relations, np.array([2, 2]))
        indices = np.flatnonzero((stars['owners'] == 0) & (stars['kinds'] == 0))
        self.assertEqual(len(indices), 1)
        members = stars['members'][stars['groups'] == indices[0]]
        self.assertEqual(members.tolist(), [0, 1, 2])
        self.assertEqual(int(stars['sizes'].sum()), len(stars['members']))
        pairwise = incidence_arrays(edges, relations, np.array([2, 2]), True)
        self.assertEqual(pairwise['sizes'].tolist(), [2, 2, 2])

    def test_equivariant_messages_are_member_specific_with_finite_gradients(self):
        hidden = torch.tensor([[1.], [2.], [3.], [9.]], requires_grad=True)
        graph = dict(members=torch.tensor([0, 1, 2]), groups=torch.zeros(3, dtype=torch.long),
                     sizes=torch.tensor([3.]))
        phi = lambda v: v
        rho = lambda v: v[:, :1] * v[:, 1:2]
        full = member_messages(hidden, graph, phi, rho, torch.zeros(1, 1))
        invariant = member_messages(hidden, graph, phi, rho, torch.zeros(1, 1), True)
        torch.testing.assert_close(full, hidden[:3] * (6 / 3**.5))
        torch.testing.assert_close(invariant, torch.full((3, 1), 12 / 3**.5))
        self.assertFalse(torch.equal(full[0], full[1]))
        full.sum().backward()
        self.assertGreater(hidden.grad[:3].abs().min().item(), 0)
        self.assertEqual(hidden.grad[3].item(), 0)

    def test_operator_is_equivariant_to_member_order_and_node_relabeling(self):
        hidden = torch.tensor([[1.], [2.], [4.], [7.]])
        graph = dict(members=torch.tensor([0, 1, 2, 1, 3]), groups=torch.tensor([0, 0, 0, 1, 1]),
                     sizes=torch.tensor([3., 2.]))
        phi = lambda v: v.square()
        rho = lambda v: (v[:, :1] - v[:, 1:2]).tanh()
        value = member_messages(hidden, graph, phi, rho, torch.zeros(2, 1))
        aggregated = segment_sum(value, graph['members'], 4)
        order = torch.tensor([4, 2, 0, 3, 1])
        permutation = torch.tensor([2, 0, 3, 1])
        inverse = torch.argsort(permutation)
        new_graph = dict(members=inverse[graph['members'][order]], groups=graph['groups'][order], sizes=graph['sizes'])
        replay = member_messages(hidden[permutation], new_graph, phi, rho, torch.zeros(2, 1))
        restored = segment_sum(replay, new_graph['members'], 4)[inverse]
        torch.testing.assert_close(aggregated, restored, atol=1e-6, rtol=1e-6)

    def test_query_fact_is_removed_before_groups_counts_and_matched_models(self):
        a = dict(node_counts=np.array([3, 2]), nd=np.array(2), pair_offset=np.array([0, 3]),
            pairs_train=np.array([[0, 1, 2], [0, 1, 1]]), y_train=np.array([1, 1, 0]),
            folds=np.array([0, 1, 0]), edges=np.array([[0, 1, 0, 1], [3, 4, 1, 2]]),
            relation=np.array([0, 0, 1, 1]), target_codes=np.array([0, 3, -1, -1]),
            x_0=np.ones((3, 2), np.float32), x_1=np.ones((2, 4), np.float32))
        graph = context(a, 0, 'cpu')
        self.assertEqual(graph['counts'][0, 0].item(), 0)
        self.assertFalse(((graph['owners'] == 0) & (graph['kinds'] < 2)).any())
        p = dict(width=8, depth=2, dropout=0., folds=2)
        states = []
        for kind in ('full', 'invariant', 'pairwise', 'no_hypergraph'):
            torch.manual_seed(7)
            model, xs, graphs, _ = setup(a, p, kind, 'cpu')
            states.append(model.state_dict())
            torch.testing.assert_close(graph['counts'], graphs[0]['counts'], atol=0, rtol=0)
            pairs = torch.tensor([[0, 1], [3, 4]])
            score = model(xs, graphs[0], pairs)
            torch.testing.assert_close(score, model(xs, graphs[0], pairs.flip(1)).flip(0))
            score.sum().backward()
            self.assertTrue(torch.isfinite(model.project[0].weight.grad).all())
        for state in states[1:]:
            for name in state:
                torch.testing.assert_close(state[name], states[0][name], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
