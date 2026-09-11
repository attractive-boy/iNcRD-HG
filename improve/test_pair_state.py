"""Checks for coordinate transport, graph masking, and model symmetries."""
import copy
import unittest

import numpy as np
import torch
from torch import nn
from torch_geometric.data import HeteroData

from improve.pair_exchange_modules import remove_target_pairs
from improve.pair_state_modules import AxisOperator, PairStateBranch, PairStateModel


def toy_graph():
    torch.manual_seed(71)
    graph = HeteroData()
    for t, n in [('lncRNA', 3), ('miRNA', 2), ('drug', 3)]:
        graph[t].x = torch.randn(n, 6)
    entries = {
        ('lncRNA', 'LncLnc', 'lncRNA'): [[0, 1, 2, 0, 1], [0, 1, 2, 1, 2]],
        ('miRNA', 'MiMi', 'miRNA'): [[0, 1, 0], [0, 1, 1]],
        ('lncRNA', 'LncMi', 'miRNA'): [[0, 1], [0, 1]],
        ('miRNA', 'rev_LncMi', 'lncRNA'): [[0, 1], [0, 1]],
        ('drug', 'DrugDrug', 'drug'): [[0, 1, 2], [0, 1, 2]],
        ('lncRNA', 'LncDrug', 'drug'): [[0, 1], [0, 1]],
        ('drug', 'rev_LncDrug', 'lncRNA'): [[0, 1], [0, 1]],
        ('miRNA', 'MiDrug', 'drug'): [[0], [2]],
        ('drug', 'rev_MiDrug', 'miRNA'): [[2], [0]],
    }
    for k, v in entries.items():
        graph[k].edge_index = torch.tensor(v, dtype=torch.long)
    return graph


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({t: nn.Linear(6, 6) for t in ('lncRNA', 'miRNA', 'drug')})

    def forward(self, x, edges, weights):
        return {t: self.layers[t](v) for t, v in x.items()}


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(12, 1)

    def forward(self, z, pairs):
        r, d = pairs
        return self.linear(torch.cat([z['lncRNA'][r], z['drug'][d]], -1)).flatten()


class ToyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.decoder = ToyEncoder(), ToyDecoder()


class PairStateTests(unittest.TestCase):
    def test_axes_equal_explicit_product_graph(self):
        r_op = AxisOperator(torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]]), 3, 3)
        d_op = AxisOperator(torch.tensor([[0, 1], [1, 0]]), 2, 2)
        field = torch.randn(3, 2, 4, requires_grad=True)
        actual = r_op(field) + d_op(field.transpose(0, 1)).transpose(0, 1)
        r = torch.sparse_coo_tensor(r_op.indices, r_op.values, r_op.shape).to_dense()
        d = torch.sparse_coo_tensor(d_op.indices, d_op.values, d_op.shape).to_dense()
        product = torch.kron(r, torch.eye(2)) + torch.kron(torch.eye(3), d)
        expected = (product @ field.reshape(6, 4)).reshape(3, 2, 4)
        torch.testing.assert_close(actual, expected)
        g1 = torch.autograd.grad(actual.square().sum(), field, retain_graph=True)[0]
        g2 = torch.autograd.grad(expected.square().sum(), field)[0]
        torch.testing.assert_close(g1, g2)

    def test_rectangular_and_empty_operators(self):
        op = AxisOperator(torch.tensor([[0, 2], [1, 1]]), 3, 2)
        x = torch.randn(3, 4, 5)
        expected = torch.stack([torch.zeros(4, 5), (x[0] + x[2]) / 2])
        torch.testing.assert_close(op(x), expected)
        identity_only = AxisOperator(torch.tensor([[0, 1], [0, 1]]), 2, 2, remove_self=True)
        torch.testing.assert_close(identity_only(x[:2]), torch.zeros_like(x[:2]))

    def test_query_guard_rebuilds_boundary_each_forward(self):
        graph = toy_graph()
        branch = PairStateBranch(graph, 6, 'lncRNA', width=4)
        edges = graph.edge_index_dict
        weights = {k: torch.ones(e.size(1)) for k, e in edges.items()}
        before = branch.boundary(edges, graph['drug'].x)
        masked, _ = remove_target_pairs(edges, weights, 'LncDrug', 'lncRNA', 3,
                                         np.array([[1], [1]]))
        after = branch.boundary(masked, graph['drug'].x)
        self.assertEqual(float(before['lncRNA'][1, 1]), 1.)
        self.assertEqual(float(after['lncRNA'][1, 1]), 0.)
        self.assertEqual(float(after['lncRNA'][0, 0]), 1.)
        torch.testing.assert_close(before['miRNA'], after['miRNA'])
        self.assertNotIn((1, 1), list(map(tuple, masked[('drug', 'rev_LncDrug', 'lncRNA')].t().tolist())))
        torch.testing.assert_close(branch.boundary(edges, graph['drug'].x)['lncRNA'], before['lncRNA'])

    def test_zero_initialized_branch_preserves_base(self):
        graph, pairs = toy_graph(), torch.tensor([[0, 1, 2], [2, 0, 1]])
        base = ToyBase()
        expected = base.decoder(base.encoder(graph.x_dict, graph.edge_index_dict, {}), pairs)
        counts = []
        for arm in ('pair_local', 'pair_features', 'pair_state'):
            torch.manual_seed(9)
            model = PairStateModel(copy.deepcopy(base), graph, 6, 'lncRNA', arm, width=4)
            scores, original, correction = model(graph.x_dict, graph.edge_index_dict, pairs, {}, True)
            torch.testing.assert_close(scores, expected, rtol=0, atol=0)
            torch.testing.assert_close(original, expected, rtol=0, atol=0)
            self.assertEqual(int(torch.count_nonzero(correction)), 0)
            counts.append(sum(p.numel() for p in model.parameters()))
        self.assertEqual(len(set(counts)), 1)

    def test_support_moves_only_along_available_coordinates(self):
        graph = toy_graph()
        branch = PairStateBranch(graph, 6, 'lncRNA', width=4, depth=2)
        nn.init.normal_(branch.head[-1].weight, std=.2)
        edges = dict(graph.edge_index_dict)
        modified = dict(edges)
        modified[('lncRNA', 'LncDrug', 'drug')] = edges[('lncRNA', 'LncDrug', 'drug')][:, 1:]
        pairs = torch.tensor([[1, 1], [0, 2]])
        with torch.no_grad():
            a, b = branch(graph.x_dict, edges, pairs), branch(graph.x_dict, modified, pairs)
            local_a = branch(graph.x_dict, edges, pairs, 'no_transport')
            local_b = branch(graph.x_dict, modified, pairs, 'no_transport')
        self.assertGreater(float((a[0] - b[0]).abs()), 1e-7)
        torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
        torch.testing.assert_close(local_a, local_b, rtol=0, atol=0)

    def test_node_permutation_and_query_chunk_equivariance(self):
        graph = toy_graph()
        maps = {'lncRNA': torch.tensor([2, 0, 1]), 'miRNA': torch.tensor([1, 0]), 'drug': torch.tensor([1, 2, 0])}
        permuted = graph.clone()
        for t, permutation in maps.items():
            permuted[t].x[permutation] = graph[t].x
        for k, e in graph.edge_index_dict.items():
            permuted[k].edge_index = torch.stack([maps[k[0]][e[0]], maps[k[2]][e[1]]])
        torch.manual_seed(17)
        a = PairStateBranch(graph, 6, 'lncRNA', width=4)
        torch.manual_seed(17)
        b = PairStateBranch(permuted, 6, 'lncRNA', width=4)
        with torch.no_grad():
            a.head[-1].weight.normal_()
            for (_, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
                q.copy_(p)
        pairs = torch.tensor([[0, 1, 2, 1], [2, 0, 1, 1]])
        new_pairs = torch.stack([maps['lncRNA'][pairs[0]], maps['drug'][pairs[1]]])
        with torch.no_grad():
            actual = a(graph.x_dict, graph.edge_index_dict, pairs)
            reordered = b(permuted.x_dict, permuted.edge_index_dict, new_pairs)
            chunked = torch.cat([a(graph.x_dict, graph.edge_index_dict, p) for p in pairs.split(2, dim=1)])
        torch.testing.assert_close(actual, reordered, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual, chunked, rtol=2e-5, atol=2e-6)

    def test_joint_gradients_after_head_update(self):
        graph, pairs = toy_graph(), torch.tensor([[1, 2], [0, 2]])
        model = PairStateModel(ToyBase(), graph, 6, 'lncRNA', 'pair_state', width=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            _, _, correction = model(graph.x_dict, graph.edge_index_dict, pairs, {}, True)
            loss = (correction - torch.tensor([1., -1.])).square().mean()
            loss.backward()
            optimizer.step()
        for component in (model.branch.rna_projections, model.branch.drug_projection,
                          model.branch.blocks, model.branch.boundary_vectors, model.base.encoder):
            gradients = [p.grad for p in component.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)


if __name__ == '__main__':
    unittest.main()
