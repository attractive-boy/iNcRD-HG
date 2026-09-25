import unittest
import numpy as np
import torch

from improve.typed_sheaf_models import rotate_stalk,transport_aggregate,physical_counts,TypedSheaf


class TypedSheafTests(unittest.TestCase):
    def test_rotation_inverse_and_norm(self):
        torch.manual_seed(7)
        x=torch.randn(7,2,3,dtype=torch.float64);a=torch.randn(7,dtype=torch.float64)
        rotated=rotate_stalk(x,a)
        torch.testing.assert_close(rotate_stalk(rotated,-a),x)
        torch.testing.assert_close(rotated.square().sum((1,2)),x.square().sum((1,2)))

    def test_operator_is_symmetric_and_identity_is_normalized_adjacency(self):
        edge=torch.tensor([[0,1,1],[1,2,3]])
        angle=torch.tensor([.1,.5,-.2],dtype=torch.float64)
        eye=torch.eye(8,dtype=torch.float64).reshape(4,2,8)
        op=transport_aggregate(eye,edge,angle).reshape(8,8)
        torch.testing.assert_close(op,op.T)
        self.assertLessEqual(torch.linalg.eigvalsh(op).abs().max().item(),1.+1e-12)
        expected=torch.zeros(4,4,dtype=torch.float64)
        degree=torch.bincount(edge.flatten(),minlength=4).double()
        for s,t in edge.T:expected[s,t]=expected[t,s]=(degree[s]*degree[t]).rsqrt()
        identity=transport_aggregate(eye,edge,angle,'identity').reshape(8,8)
        torch.testing.assert_close(identity,torch.kron(expected,torch.eye(2,dtype=torch.float64)))

    def test_reversing_physical_edges_transposes_transport(self):
        h=torch.randn(4,2,5);edge=torch.tensor([[0,1,1],[1,2,3]]);a=torch.randn(3)
        torch.testing.assert_close(transport_aggregate(h,edge,a),transport_aggregate(h,edge.flip(0),-a))

    def test_counts_are_recomputed_after_edge_removal(self):
        edge=torch.tensor([[0,1,1],[1,2,3]]);rel=torch.tensor([0,1,0])
        full=physical_counts(edge,rel,4,2);masked=physical_counts(edge[:,:2],rel[:2],4,2)
        self.assertEqual(masked[3,0],0.)
        self.assertLess(masked[1,0],full[1,0])

    def test_query_labels_are_not_model_inputs_and_order_is_invariant(self):
        torch.manual_seed(9)
        model=TypedSheaf([3,4],[3,2],2,8,2,'orthogonal').eval()
        xs=[torch.randn(3,3),torch.randn(2,4)];edge=torch.tensor([[0,1,2,1],[3,3,4,2]])
        rel=torch.tensor([0,0,0,1]);q=torch.tensor([[0,2,1],[4,3,4]])
        with torch.no_grad():
            a=model(xs,edge,rel,q);b=model(xs,edge,rel,q.flip(1)).flip(0)
        torch.testing.assert_close(a,b)


if __name__=='__main__':unittest.main()
