"""Tests of the exchange objective, masking, and model-selection boundaries."""
import math
import unittest

import numpy as np
import torch
from torch_geometric.data import HeteroData

from improve.pair_exchange import inner_partition, pick_config
from improve.pair_exchange_modules import (DecomposedDecoder, auxiliary_loss,
    exchange_pairs, exchange_support, positive_matrix, remove_target_pairs,
    restrict_training_graph, sample_exchanges, sample_queries)


class PairExchangeChecks(unittest.TestCase):
    def test_additive_invariance_and_zero_marginal_gradient(self):
        torch.manual_seed(2)
        base = torch.randn(4, 17, dtype=torch.float64, requires_grad=True)
        a, b = torch.randn(2, 17, dtype=torch.float64, requires_grad=True), torch.randn(2, 17, dtype=torch.float64, requires_grad=True)
        additive = torch.stack([a[0]+b[0], a[1]+b[1], a[0]+b[1], a[1]+b[0]])
        loss = auxiliary_loss(base+additive, 'exchange')
        torch.testing.assert_close(loss, auxiliary_loss(base, 'exchange'), rtol=1e-12, atol=1e-12)
        ga, gb = torch.autograd.grad(loss, (a,b))
        torch.testing.assert_close(ga, torch.zeros_like(ga), rtol=0, atol=1e-15)
        torch.testing.assert_close(gb, torch.zeros_like(gb), rtol=0, atol=1e-15)
        self.assertAlmostEqual(float(auxiliary_loss(additive, 'exchange')), math.log(2))

    def test_pairwise_is_distinct_and_swap_orientation_is_correct(self):
        preferred = torch.tensor([[2.],[2.],[-2.],[-2.]], requires_grad=True)
        self.assertLess(float(auxiliary_loss(preferred, 'exchange')), .03)
        self.assertGreater(float(auxiliary_loss(-preferred, 'exchange')), 3.9)
        additive = torch.tensor([[5.],[-5.],[5.],[-5.]])
        self.assertGreater(float(auxiliary_loss(additive, 'pairwise')), math.log(2))
        self.assertAlmostEqual(float(auxiliary_loss(additive, 'exchange')), math.log(2), places=6)

    def test_sampler_preserves_endpoints_and_excludes_fitting_positives(self):
        p = np.array([[0,1,2,3],[0,1,2,3]])
        known = positive_matrix(p, np.ones(4), 4, 4)
        known[0,1] = True
        exchanges = sample_exchanges(p, known, 100, 73)
        np.testing.assert_array_equal(exchanges, sample_exchanges(p, known, 100, 73))
        self.assertEqual(exchanges.shape, (4,100))
        indices = exchange_pairs(exchanges).reshape(2,4,-1)
        self.assertTrue(known[tuple(indices[:,0])].all())
        self.assertTrue(known[tuple(indices[:,1])].all())
        self.assertFalse(known[tuple(indices[:,2])].any())
        self.assertFalse(known[tuple(indices[:,3])].any())
        np.testing.assert_array_equal(np.sort(indices[:,:2],axis=1), np.sort(indices[:,2:],axis=1))

    def test_degenerate_exchange_is_bounded_and_has_zero_differentiable_loss(self):
        p = np.array([[0,1,2],[0,0,0]])
        known = positive_matrix(p,np.ones(3),3,2)
        swaps = sample_exchanges(p,known,100,1)
        self.assertEqual(swaps.shape,(4,0))
        values = torch.empty((4,0),requires_grad=True)
        loss = auxiliary_loss(values,'exchange')
        loss.backward()
        self.assertEqual(float(loss),0.)
        self.assertEqual(values.grad.shape,values.shape)

    def test_query_rng_is_stratified_reproducible_and_has_no_global_side_effect(self):
        y = np.r_[np.ones(100),np.zeros(200)]
        np.random.seed(49)
        expected = np.random.rand()
        np.random.seed(49)
        selected = sample_queries(y,.5,6)
        self.assertEqual(np.random.rand(),expected)
        self.assertEqual(len(selected),150)
        self.assertEqual(int(y[selected].sum()),50)
        np.testing.assert_array_equal(selected,sample_queries(y,.5,6))

    def toy_graph(self):
        graph = HeteroData()
        graph['rna'].x = torch.randn(4,3)
        graph['drug'].x = torch.randn(3,2)
        graph['rna','RDrug','drug'].edge_index = torch.tensor([[0,1,2,3],[0,1,2,1]])
        graph['drug','rev_RDrug','rna'].edge_index = graph['rna','RDrug','drug'].edge_index.flip(0).clone()
        graph['rna','sim','rna'].edge_index = torch.tensor([[0,1,2],[1,2,3]])
        weights = {k:torch.arange(v.size(1),dtype=torch.float)+1 for k,v in graph.edge_index_dict.items()}
        return graph, weights

    def test_query_mask_removes_both_directions_and_preserves_other_relations(self):
        graph, weights = self.toy_graph()
        original = {k:v.clone() for k,v in graph.edge_index_dict.items()}
        held = np.array([[1,3],[1,1]])
        edges, actual_weights = remove_target_pairs(graph.edge_index_dict,weights,'RDrug','rna',3,held)
        key, reverse = ('rna','RDrug','drug'),('drug','rev_RDrug','rna')
        torch.testing.assert_close(edges[key],torch.tensor([[0,2],[0,2]]))
        torch.testing.assert_close(edges[reverse],edges[key].flip(0))
        torch.testing.assert_close(actual_weights[key],torch.tensor([1.,3.]))
        torch.testing.assert_close(edges['rna','sim','rna'],original['rna','sim','rna'])
        for k in original:
            torch.testing.assert_close(graph[k].edge_index,original[k])

    def test_inner_support_restricts_to_fitting_labels(self):
        graph, weights = self.toy_graph()
        pairs = np.array([[0,2,0,3],[0,2,2,0]])
        labels = np.array([1.,1.,0.,0.],dtype=np.float32)
        restricted, new_weights = restrict_training_graph(graph,weights,'RDrug','rna',pairs,labels)
        key = ('rna','RDrug','drug')
        np.testing.assert_array_equal(restricted[key].edge_index.numpy(),np.array([[0,2],[0,2]]))
        self.assertEqual(graph[key].edge_index.shape[1],4)
        torch.testing.assert_close(new_weights[key],torch.tensor([1.,3.]))

    def test_interaction_has_zero_training_row_and_column_means(self):
        torch.manual_seed(19)
        model = DecomposedDecoder(6,'rna',[0,1,3],[0,2],rank=3).double()
        z = dict(rna=torch.randn(5,6,dtype=torch.float64),drug=torch.randn(4,6,dtype=torch.float64))
        a,b,u,v = model.tables(z)
        surface = u @ v.T / math.sqrt(3)
        torch.testing.assert_close(surface[[0,1,3]].mean(0),torch.zeros(4,dtype=torch.float64),atol=1e-12,rtol=0)
        torch.testing.assert_close(surface[:,[0,2]].mean(1),torch.zeros(5,dtype=torch.float64),atol=1e-12,rtol=0)
        pairs = torch.tensor([[0,3,4],[1,2,3]])
        marginal, interaction = model.components(z,pairs)
        torch.testing.assert_close(model(z,pairs),marginal+interaction)
        torch.testing.assert_close(interaction,surface[pairs[0],pairs[1]])

    def test_inner_partition_exhaustive_disjoint_and_stratified(self):
        labels = np.r_[np.ones(100),np.zeros(200)]
        fit, validation = inner_partition(labels)
        self.assertFalse(set(fit).intersection(validation))
        np.testing.assert_array_equal(np.sort(np.r_[fit,validation]),np.arange(len(labels)))
        self.assertEqual(int(labels[validation].sum()),20)
        self.assertEqual(len(validation),60)

    def test_selection_rejects_outer_rows_and_resolves_ties_once(self):
        rows = [dict(stage='inner',config_id='c2',criterion=.9),dict(stage='inner',config_id='c1',criterion=.9)]
        self.assertEqual(pick_config(rows)['config_id'],'c1')
        with self.assertRaises(ValueError):
            pick_config([dict(stage='outer',config_id='c0',criterion=1.)])

    def test_support_audit_counts_a_single_observed_tetrad(self):
        pairs = np.array([[0,1,0,1],[0,1,1,0]])
        audit = exchange_support(pairs,np.array([1,1,0,0]),2,2)
        self.assertEqual(audit,dict(positive_edges=2,legal_unlabeled_swaps=1,observed_negative_swaps=1,covered_positive_edges=2))


if __name__ == '__main__':
    unittest.main()
