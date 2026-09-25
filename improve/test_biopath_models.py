import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F

from improve.biopath_models import PathContext,PathConv,PathResidual,context,degree_preserving_shuffle


def scatter_add(src,index,dim=0,dim_size=None):
    shape=list(src.shape);shape[dim]=dim_size
    return src.new_zeros(shape).index_add(dim,index,src)


def scatter_mean(src,index,dim=0,dim_size=None):
    count=torch.bincount(index,minlength=dim_size).clamp_min(1)
    return scatter_add(src,index,dim,dim_size)/count[:,None,None]


def extremum(src,index,dim,dim_size,mode):
    shape=list(src.shape);shape[dim]=dim_size
    initial=-float('inf') if mode=='amax' else float('inf')
    result=src.new_full(shape,initial).scatter_reduce(dim,index[:,None,None].expand_as(src),src,reduce=mode)
    return result,None


class BioPathTests(unittest.TestCase):
    def test_pna_port_matches_frozen_author_equations(self):
        path=Path(__file__).parent/'research/path_geometry_20260914/author_biopathnet_layer.py'
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='GeneralizedRelationalConv')
        ns=dict(torch=torch,nn=nn,F=F,layers=SimpleNamespace(MessagePassingBase=nn.Module),
                scatter_add=scatter_add,scatter_mean=scatter_mean,
                scatter_max=lambda src,index,dim,dim_size:extremum(src,index,dim,dim_size,'amax'),
                scatter_min=lambda src,index,dim,dim_size:extremum(src,index,dim,dim_size,'amin'))
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
        torch.manual_seed(3);width=4;relations=3;nodes=5
        reference=ns['GeneralizedRelationalConv'](width,width,relations,width,layer_norm=True)
        port=PathConv(width,relations);port.load_state_dict(reference.state_dict(),strict=True)
        edge=torch.tensor([[0,1,1,2,3,4],[1,0,2,3,4,3]])
        rel=torch.tensor([0,0,1,2,1,2]);degree=torch.bincount(edge[1],minlength=nodes).float()
        q=torch.randn(2,width);h=torch.randn(nodes,2,width);boundary=torch.zeros_like(h)
        boundary[0,0]=q[0];boundary[2,1]=q[1]
        graph=SimpleNamespace(query=q,boundary=boundary,edge_list=torch.cat([edge.T,rel[:,None]],1),
                              num_relation=relations,num_node=nodes,degree_out=degree,
                              edge_weight=torch.ones(len(rel)),device=torch.device('cpu'))
        expected=reference.combine(h,reference.aggregate(graph,reference.message(graph,h)))
        actual=port(PathContext(edge,rel,degree+1,nodes),h,boundary,q)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)

    def test_switch_control_keeps_each_endpoint_degree(self):
        m=sparse.csr_matrix((np.ones(10),(np.array([0,0,1,1,2,2,3,3,4,4]),np.array([0,1,1,2,2,3,3,4,4,0]))),shape=(5,5))
        z,record=degree_preserving_shuffle(m,13,20)
        np.testing.assert_array_equal(np.asarray(z.sum(0)),np.asarray(m.sum(0)))
        np.testing.assert_array_equal(np.asarray(z.sum(1)),np.asarray(m.sum(1)))
        self.assertEqual(z.nnz,m.nnz);self.assertGreater(record['changed_fraction'],0.)
        self.assertTrue(np.all(z.data==1))

    def test_query_removal_also_removes_its_reverse(self):
        a=dict(pairs_train=np.array([[0,1],[0,1]]),folds=np.array([0,1]),nd=np.array(2),
               edges=np.array([[0,1,0],[2,3,1]]),shuffled_edges=np.array([[0,1,0],[2,3,1]]),
               relation=np.array([0,0,1]),target_codes=np.array([0,3,-1]),background=np.zeros(3,dtype=bool),
               symmetric_relations=np.array([False,True]),physical_relations=np.array(2),base_nodes=np.array(4),nodes=np.array(4))
        c=context(a,0,'full','cpu')
        edges=set(map(tuple,c.edge.T.numpy()))
        self.assertNotIn((0,2),edges);self.assertNotIn((2,0),edges)
        self.assertIn((1,3),edges);self.assertIn((3,1),edges)
        symmetric=c.relation[(c.edge[0]<2)&(c.edge[1]<2)]
        torch.testing.assert_close(symmetric,torch.tensor([1,1]))

    def test_query_order_and_batch_do_not_change_path_scores(self):
        torch.manual_seed(7)
        edge=torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]])
        c=PathContext(edge,torch.zeros(6,dtype=torch.long),torch.bincount(edge[1],minlength=4).float()+1,4)
        model=PathResidual(2,4,2).eval();nn.init.normal_(model.mlp[-1].weight,std=.1)
        with torch.no_grad():
            full=model(c,torch.tensor([0,3]),0,0,4)
            single=torch.cat([model(c,torch.tensor([j]),0,0,4) for j in (0,3)])
            reverse=model(c,torch.tensor([3,0]),0,0,4).flip(0)
        torch.testing.assert_close(full,single,atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(full,reverse,atol=2e-6,rtol=2e-6)


if __name__=='__main__':unittest.main()
