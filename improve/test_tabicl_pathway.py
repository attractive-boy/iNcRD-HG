import gzip
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy import sparse

from improve.rna_pathway_data import inspect_encori
from improve.tabicl_pathway_modules import ProgramPairs, assess
from improve import tabicl_pathway


class PathwayEvidenceTests(unittest.TestCase):
    def data(self):
        gp = sparse.csr_matrix([[1,0],[1,1],[0,1],[0,0]],dtype=float)
        dg = sparse.csr_matrix([[1,0,0,0],[0,1,1,0],[0,0,0,0]],dtype=float)
        rg = sparse.csr_matrix([[1,1,0,0],[0,0,1,0],[0,0,0,0]],dtype=float)
        ng = sparse.csr_matrix([[0,0,.5,0],[.4,.6,0,0],[0,0,0,0]],dtype=float)
        return gp,dg,rg,ng

    def model(self):
        gp,dg,rg,ng = self.data()
        return ProgramPairs(gp,dg,{'MiDrug':rg,'LncDrug':rg},{'MiDrug':ng,'LncDrug':ng})

    def test_same_rule_for_both_task_names(self):
        m=self.model()
        np.testing.assert_array_equal(m.table('MiDrug','cross_rna'),m.table('LncDrug','cross_rna'))

    def test_gene_and_pathway_reordering_preserves_scores(self):
        gp,dg,rg,ng=self.data()
        perm=[2,0,3,1]
        other=ProgramPairs(gp[perm][:,[1,0]],dg[:,perm],{'MiDrug':rg[:,perm]}, {'MiDrug':ng[:,perm]})
        np.testing.assert_allclose(self.model().table('MiDrug','cross_rna'),other.table('MiDrug','cross_rna'),atol=1e-7)

    def test_missing_evidence_is_finite_and_has_zero_matches(self):
        table=self.model().table('MiDrug','direct')
        self.assertTrue(np.isfinite(table).all())
        np.testing.assert_array_equal(table[2,:,[0,2,4,5,6,7,8,10,11,12,13]],0.)
        np.testing.assert_array_equal(table[:,2,[1,3,4,5,6,7,9,10,11,12,13]],0.)

    def test_query_order_and_duplicates_are_pure_lookup(self):
        m=self.model()
        pairs=np.array([[0,1,0,2],[1,0,1,2]])
        result=m.transform('MiDrug',pairs,'cross_rna')
        np.testing.assert_array_equal(result[::-1],m.transform('MiDrug',pairs[:,::-1],'cross_rna'))
        np.testing.assert_array_equal(result[0],result[2])
        with self.assertRaises(ValueError):m.transform('MiDrug',np.array([[-1],[0]]),'direct')

    def test_http_200_php_errors_are_not_valid_data(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'sample.gz'
            with gzip.open(p,'wt') as f:f.write('geneID\tgeneName\nENSG1\tABC\nConnect to database fail!\n')
            with self.assertRaises(ValueError):inspect_encori(p,'encori_lnc')
            with gzip.open(p,'wt') as f:f.write('geneID\tgeneName\nThe "RNA" parameter haven\'t been set correctly! Or the input of "RNA" parameter is not available!\n')
            self.assertEqual(inspect_encori(p,'encori_lnc')['query_status'],'unavailable_name_or_no_annotation')

    def test_tiny_lnc_decline_fails_and_macro_ap_is_separate(self):
        config=dict(id='candidate',variant='direct',alpha=.1)
        rows=[]
        for task in ['MiDrug','LncDrug']:
            for s in [666,0,1]:
                rows.extend([dict(task=task,seed=s,recipe='baseline',auc=.8,ap=.7,macro_drug_ap=.7),
                             dict(task=task,seed=s,recipe='candidate',auc=.81 if task=='MiDrug' else .8,
                                  ap=.71 if task=='MiDrug' else .7,macro_drug_ap=.5)])
        self.assertTrue(assess(rows,config,[666,0,1])['eligible'])
        rows[-1]['ap']-=1e-10
        self.assertFalse(assess(rows,config,[666,0,1])['eligible'])

    def test_outer_gate_precedes_historical_array_access(self):
        with mock.patch.object(tabicl_pathway,'selected',side_effect=ValueError('no eligible result')):
            with mock.patch.object(tabicl_pathway.graph,'protocol') as loader:
                with self.assertRaises(ValueError):
                    tabicl_pathway.stage_data('/unused',{'tasks':['MiDrug']},'MiDrug','outer')
                loader.assert_not_called()

    def test_fold_context_alignment_and_query_label_independence(self):
        p={'input_configs':{'MiDrug':{'input_view':'raw'}},
           'graph_recipes':{'MiDrug':{'arm':'fold_match'}}}
        a=dict(raw_train=np.zeros((5,36)),raw_val=np.zeros((2,36)),
               raw_query_views=np.arange(5)[:,None,None]*np.ones((5,2,36)),
               fold_ids=np.arange(5),y_train=np.array([1,0,1,0,1]),y_val=np.array([1,0]))
        extra=np.arange(5)[:,None].astype(float)
        query=np.array([[9.],[8.]])
        x,y,q=tabicl_pathway.job_arrays(p,'MiDrug',a,'fold2',extra,query)
        self.assertEqual(x[0,-1],2.)
        self.assertEqual(y.tolist(),[1])
        np.testing.assert_array_equal(q[0,:,:36],2.)
        np.testing.assert_array_equal(q[0,:,-1],query[:,0])
        a['y_val']=1-a['y_val']
        _,_,after=tabicl_pathway.job_arrays(p,'MiDrug',a,'fold2',extra,query)
        np.testing.assert_array_equal(q,after)


if __name__ == '__main__':
    unittest.main()
