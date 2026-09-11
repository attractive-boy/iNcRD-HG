import unittest

import numpy as np
import torch
from torch_geometric.data import HeteroData

from improve.graph_tabm_modules import BatchEnsembleLinear, GraphTabM, PairFeatureEncoder, crossfit_features, fit_transform


def graph_fixture():
    torch.manual_seed(3)
    g = HeteroData()
    for nt,n in [('lncRNA',8),('miRNA',5),('drug',6)]:
        g[nt].x = torch.randn(n,7)
    g['lncRNA','LncLnc','lncRNA'].edge_index = torch.tensor([[0,1,2,3,4,5,6,7],[1,2,3,4,5,6,7,0]])
    g['miRNA','MiMi','miRNA'].edge_index = torch.tensor([[0,1,2,3,4],[1,2,3,4,0]])
    g['drug','DrugDrug','drug'].edge_index = torch.tensor([[0,1,2,3,4,5],[1,2,3,4,5,0]])
    g['lncRNA','LncMi','miRNA'].edge_index = torch.tensor([[0,1,2,3,4],[0,1,2,3,4]])
    g['miRNA','MiDrug','drug'].edge_index = torch.tensor([[0,1,2,3,4],[0,1,2,3,4]])
    pairs = np.array([[i for i in range(8) for _ in range(3)], [j for i in range(8) for j in range(3)]])
    labels = np.array([1,0,0]*8, dtype=np.float32)
    pos = torch.from_numpy(pairs[:, labels==1])
    g['lncRNA','LncDrug','drug'].edge_index = pos
    g['drug','rev_LncDrug','lncRNA'].edge_index = pos.flip(0)
    return g,pairs,labels


class TabMTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_batchensemble_matches_explicit_member_matrices(self):
        layer = BatchEnsembleLinear(4,3,5,first=True)
        x = torch.randn(7,5,4)
        expected = torch.stack([x[:,k] @ (layer.weight*layer.r[k][None,:]*layer.s[k][:,None]).T + layer.bias[k]
                                for k in range(5)],1)
        torch.testing.assert_close(layer(x),expected)

    def test_feature_encoder_rejects_self_label_and_crossfit_excludes_it(self):
        g,pairs,y = graph_fixture()
        encoder = PairFeatureEncoder(g,'LncDrug')
        with self.assertRaises(ValueError):
            encoder.transform(pairs, pairs[:,y==1])
        val = np.array([[0,1,2],[3,4,5]])
        x,xv,names,fold = crossfit_features(g,'LncDrug',pairs,y,val,folds=2)
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(x.shape[1],len(names))
        for k in np.unique(fold):
            expected = encoder.transform(pairs[:,fold==k],pairs[:,(fold!=k)&(y==1)]).values
            np.testing.assert_array_equal(x[fold==k],expected)
        np.testing.assert_array_equal(xv,encoder.transform(val,pairs[:,y==1]).values)

    def test_preprocessing_fits_training_only(self):
        train = np.arange(40,dtype=np.float32).reshape(20,2)
        a,_,f = fit_transform(train,np.zeros((3,2),dtype=np.float32))
        b,_,g = fit_transform(train,np.full((3,2),1e6,dtype=np.float32))
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(f.quantiles_,g.quantiles_)

    def test_individual_member_loss_has_finite_gradients_and_batch_invariance(self):
        model = GraphTabM(6,width=16,depth=2,members=4,dropout=0.)
        x,y = torch.randn(9,6),torch.tensor([0.,1.,0.]*3)
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits,y[:,None].expand_as(logits))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue((model.output_weight.grad.abs().sum(1)>0).all())
        model.eval()
        torch.testing.assert_close(model(x),torch.cat([model(x[:4]),model(x[4:])]))


if __name__=='__main__':
    unittest.main()
