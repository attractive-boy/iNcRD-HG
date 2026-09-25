"""Query removal and anonymous-vocabulary invariance, beyond metric checks."""
import unittest

import numpy as np
import torch

from improve.reification_transfer import context_graph, facts_from_physical
from improve.vendor.reifm_20260914.models import ReiFM
from improve.vendor.reifm_20260914.reify import reify


class ReificationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(23)
        self.model=ReiFM(dim=8,num_layers=3).eval()

    def test_query_fact_removed_before_all_six_connections(self):
        a=dict(edges=np.array([[0,1,2],[3,3,4]]),relation=np.array([0,0,1]),
               target_codes=np.array([0,1,-1]),num_entities=np.array(5),num_relations=np.array(2))
        g=context_graph(a,[1])
        self.assertEqual(g.num_facts,2)
        self.assertEqual(g.edge_index.shape[1],12)
        self.assertFalse(bool(((g.fact_head==1)&(g.fact_tail==3)).any()))
        self.assertEqual(g.rel_offset,7)

    def test_symmetric_fact_has_both_orientations(self):
        e,r,c=facts_from_physical(np.array([[0,1],[1,2]]),np.array([0,1]),np.array([-1,12]),
                                  [('RNA','similar','RNA'),('RNA','target','drug')])
        self.assertEqual(set(map(tuple,e.T)),{(0,1),(1,0),(1,2)})
        self.assertEqual(int((c==12).sum()),1)

    def test_entity_and_relation_relabeling_are_equivariant(self):
        e=torch.tensor([[0,1,2,3],[1,2,3,4]])
        r=torch.tensor([0,0,1,1])
        g=reify(e,r,5,2)
        ep=torch.tensor([3,1,4,0,2])
        rp=torch.tensor([1,0])
        moved=reify(ep[e],rp[r],5,2)
        with torch.no_grad():
            first=self.model(g,torch.tensor([0]),torch.tensor([0]),torch.tensor([0]))
            second=self.model(moved,ep[torch.tensor([0])],rp[torch.tensor([0])],torch.tensor([0]))
        torch.testing.assert_close(first,second[:,ep],atol=2e-6,rtol=2e-6)


if __name__=='__main__':
    unittest.main()
