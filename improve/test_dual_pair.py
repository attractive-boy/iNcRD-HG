"""Equal-capacity dual-path controls and dependence on both paths."""
import copy
import unittest

import torch
from torch import nn

from improve.dual_pair_modules import DualPairBranch,DualPairModel
from improve.test_pair_state import ToyBase,toy_graph


class DualPairTests(unittest.TestCase):
    def test_equal_initial_parameters_and_original_function(self):
        graph=toy_graph()
        pairs=torch.tensor([[0,1,2],[2,0,1]])
        base=ToyBase()
        expected=base.decoder(base.encoder(graph.x_dict,graph.edge_index_dict,{}),pairs)
        states=[]
        for arm in ('dual_local','dual_transport','dual_pair'):
            torch.manual_seed(29)
            model=DualPairModel(copy.deepcopy(base),graph,6,'lncRNA',arm,width=4)
            states.append(model.state_dict())
            torch.testing.assert_close(model(graph.x_dict,graph.edge_index_dict,pairs,{}),expected,atol=0,rtol=0)
        for key in states[0]:
            for other in states[1:]:
                torch.testing.assert_close(states[0][key],other[key],atol=0,rtol=0)

    def test_disabling_transport_equals_matched_local_control(self):
        graph=toy_graph()
        pairs=torch.tensor([[0,1,2],[2,0,1]])
        model=DualPairBranch(graph,6,'lncRNA',width=4)
        nn.init.normal_(model.head[-1].weight)
        control=DualPairBranch(graph,6,'lncRNA',mode='dual_local',width=4)
        control.load_state_dict(model.state_dict())
        a=model(graph.x_dict,graph.edge_index_dict,pairs,'no_transport')
        b=control(graph.x_dict,graph.edge_index_dict,pairs)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertFalse(torch.allclose(a,model(graph.x_dict,graph.edge_index_dict,pairs)))

    def test_both_streams_receive_joint_gradients(self):
        graph=toy_graph()
        pairs=torch.tensor([[0,1,2],[2,0,1]])
        model=DualPairBranch(graph,6,'lncRNA',width=4)
        nn.init.normal_(model.head[-1].weight)
        for condition in (None,'zero_local'):
            model.zero_grad(set_to_none=True)
            model(graph.x_dict,graph.edge_index_dict,pairs,condition).square().mean().backward()
            own=sum(float(p.grad.abs().sum()) for p in model.local_blocks.parameters() if p.grad is not None)
            transported=sum(float(p.grad.abs().sum()) for p in model.blocks.parameters() if p.grad is not None)
            if condition is None:
                self.assertGreater(own,0.)
            else:
                self.assertEqual(own,0.)
            self.assertGreater(transported,0.)
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))


if __name__=='__main__':
    unittest.main()
