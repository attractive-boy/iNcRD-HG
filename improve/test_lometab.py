import unittest

import numpy as np
import torch
from torch_geometric.data import HeteroData

from improve.graph_tabm_modules import GraphTabM, PairFeatureEncoder, crossfit_features, fit_transform
from improve.lometab_modules import LoMETab, LoMETabLinear, group_ranking_loss


def graph_fixture():
    torch.manual_seed(3)
    g = HeteroData()
    for nt, n in [('lncRNA', 8), ('miRNA', 5), ('drug', 6)]:
        g[nt].x = torch.randn(n, 7)
    g['lncRNA', 'LncLnc', 'lncRNA'].edge_index = torch.tensor([[0,1,2,3,4,5,6,7],[1,2,3,4,5,6,7,0]])
    g['miRNA', 'MiMi', 'miRNA'].edge_index = torch.tensor([[0,1,2,3,4],[1,2,3,4,0]])
    g['drug', 'DrugDrug', 'drug'].edge_index = torch.tensor([[0,1,2,3,4,5],[1,2,3,4,5,0]])
    g['lncRNA', 'LncMi', 'miRNA'].edge_index = torch.tensor([[0,1,2,3,4],[0,1,2,3,4]])
    g['miRNA', 'MiDrug', 'drug'].edge_index = torch.tensor([[0,1,2,3,4],[0,1,2,3,4]])
    pairs = np.array([[i for i in range(8) for _ in range(3)], [j for i in range(8) for j in range(3)]])
    labels = np.array([1, 0, 0] * 8, dtype=np.float32)
    pos = torch.from_numpy(pairs[:, labels == 1])
    g['lncRNA', 'LncDrug', 'drug'].edge_index = pos
    g['drug', 'rev_LncDrug', 'lncRNA'].edge_index = pos.flip(0)
    return g, pairs, labels


class LoMETabTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_lometab_linear_matches_explicit_member_matrices(self):
        layer = LoMETabLinear(4, 3, members=5, rank=2, sigma_init=.1)
        x = torch.randn(7, 5, 4)
        expected = []
        for k in range(5):
            ab = layer.A[k] @ layer.B[k].T
            effective = layer.weight * (1.0 + ab)
            expected.append(x[:, k] @ effective.T + layer.bias[k])
        torch.testing.assert_close(layer(x), torch.stack(expected, 1))

    def test_sigma_init_zero_zeroes_rank_factors(self):
        layer = LoMETabLinear(6, 16, members=4, rank=2, sigma_init=0.)
        self.assertTrue(torch.all(layer.A == 0))
        self.assertTrue(torch.all(layer.B == 0))
        self.assertTrue(layer.weight.abs().sum() > 0)
        layer2 = LoMETabLinear(6, 16, members=4, rank=2, sigma_init=.1)
        self.assertTrue(layer2.A.abs().sum() > 0)

    def test_lometab_batch_invariance_and_shape(self):
        model = LoMETab(6, width=16, depth=2, members=4, rank=2, sigma_init=.1, dropout=0.)
        x = torch.randn(9, 6)
        out = model(x)
        self.assertEqual(out.shape, (9, 4))
        model.eval()
        torch.testing.assert_close(model(x), torch.cat([model(x[:4]), model(x[4:])]))

    def test_finite_gradients_through_group_loss(self):
        model = LoMETab(6, width=16, depth=2, members=4, rank=2, sigma_init=.1, dropout=0.)
        x = torch.randn(9, 6)
        y = torch.tensor([0., 1., 0.] * 3)
        pairs = np.array([[0, 1, 2, 0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 1, 2, 2, 2]])
        logits = model(x)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[:, None].expand_as(logits))
        group = group_ranking_loss(logits, y, pairs)
        (bce + group).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(bool(group.item()) > 0.)


class GroupRankingLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_hand_computed_value(self):
        logits = torch.tensor([[0., 0.], [2., 2.], [1., 1.]])
        targets = torch.tensor([0., 1., 0.])
        pairs = np.array([[0, 1, 2], [0, 0, 0]])  # drug 0 rows, one pos (row1), two neg (row0, row2)
        loss = group_ranking_loss(logits, targets, pairs)
        # within drug 0: pairs (pos=1) x (neg=0, 2) over M=2 members:
        # softplus(-2) twice and softplus(-1) twice, averaged over all elements.
        expected = (torch.nn.functional.softplus(torch.tensor(-2.0)) +
                    torch.nn.functional.softplus(torch.tensor(-1.0))) / 2
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_permutation_invariance(self):
        logits = torch.randn(12, 3)
        targets = torch.tensor([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1])
        pairs = np.array([[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]])
        order = np.random.RandomState(0).permutation(12)
        a = group_ranking_loss(logits, targets, pairs)
        b = group_ranking_loss(logits[order], targets[order], pairs[:, order])
        self.assertAlmostEqual(float(a), float(b), places=6)

    def test_no_drug_with_both_classes_returns_zero(self):
        logits = torch.randn(6, 2, requires_grad=True)
        targets = torch.tensor([1., 1., 1., 1., 1., 1.])
        pairs = np.array([[0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 1]])
        loss = group_ranking_loss(logits, targets, pairs)
        self.assertEqual(float(loss), 0.)
        self.assertFalse(loss.requires_grad)

    def test_ranked_positive_above_negative_gives_smaller_loss(self):
        pairs = np.array([[0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 1]])
        targets = torch.tensor([1., 0., 1., 1., 0., 1.])
        good = torch.tensor([[3., 3.], [-3., -3.], [2., 2.], [2., 2.], [-2., -2.], [1., 1.]])
        bad = torch.tensor([[-3., -3.], [3., 3.], [-2., -2.], [-2., -2.], [2., 2.], [-1., -1.]])
        self.assertLess(float(group_ranking_loss(good, targets, pairs)),
                        float(group_ranking_loss(bad, targets, pairs)))

    def test_gradient_flow_into_member_logits(self):
        logits = torch.randn(8, 4, requires_grad=True)
        targets = torch.tensor([1., 0., 1., 0., 1., 0., 1., 0.])
        pairs = np.array([[0, 1, 2, 0, 1, 2, 0, 1], [0, 0, 0, 1, 1, 1, 2, 2]])
        loss = group_ranking_loss(logits, targets, pairs)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad.abs().sum() > 0))


class ReusedPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_feature_encoder_and_crossfit_still_work(self):
        g, pairs, y = graph_fixture()
        val = np.array([[0, 1, 2], [3, 4, 5]])
        x, xv, names, fold = crossfit_features(g, 'LncDrug', pairs, y, val, folds=2)
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(x.shape[1], len(names))
        self.assertEqual(x.shape[1], 34)


if __name__ == '__main__':
    unittest.main()