"""Tests of evidence semantics, matched shuffles and projection boundaries."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
from scipy import sparse

from improve import site_evidence_features as f


class SiteEvidenceTests(unittest.TestCase):
    def test_exact_evidence_tokens_and_conflicts(self):
        self.assertEqual(f.evidence_class('Non-Functional MTI'),2)
        self.assertEqual(f.evidence_class('Non-Functional MTI (Weak)'),2)
        self.assertEqual(f.evidence_class('Functional MTI (Weak)'),1)
        self.assertEqual(f.evidence_class('Functional MTI;Functional MTI (Weak)'),0)
        self.assertEqual(f.evidence_class('Functional MTI (Weak);Non-Functional MTI'),3)
        with self.assertRaises(ValueError):
            f.evidence_class('unverified assay')

    def test_pool_mean_sd_gene_counts_and_empty_channels(self):
        x = np.tile(np.asarray([1,3,9],float)[:,None],(1,30))
        t = f.pool(x,[7,7,8],[0,0,1],['a','a','b'],4)
        z = t['payload'].reshape(2,4,63)
        np.testing.assert_array_equal(t['ids'],[7,8])
        np.testing.assert_array_equal(t['gene_counts'],[1,1])
        np.testing.assert_allclose(z[0,0,:30],2)
        np.testing.assert_allclose(z[0,0,30:60],1)
        np.testing.assert_allclose(z[0,0,60:],[1,np.log(3),np.log(2)])
        np.testing.assert_array_equal(z[0,1:],0)

    def test_gene_shuffle_changes_local_assignment_not_rna_moments_or_topology(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(8,30))
        rna = np.asarray([0]*4+[1]*4)
        category = np.asarray([0]*4+[1]*4)
        genes = np.asarray(['a','a','b','b','a','b','b','b'])
        permutation,audit = f.permute_groups(np.column_stack([rna,category]),genes)
        np.testing.assert_array_equal(rna[permutation],rna)
        np.testing.assert_array_equal(category[permutation],category)
        self.assertEqual(audit['identity_changed'],6)
        original = f.pool(x,rna,category,genes,4)
        changed = f.pool(x[permutation],rna,category,genes,4)
        np.testing.assert_allclose(original['payload'],changed['payload'],atol=1e-6)
        np.testing.assert_array_equal(original['counts'],changed['counts'])
        selected = genes=='a'
        self.assertFalse(np.allclose(x[selected].mean(0),x[permutation][selected].mean(0)))

    def test_evidence_shuffle_cannot_move_query_rna_vectors_into_support(self):
        rna = np.repeat(np.arange(6),3)
        support = rna<3
        category = np.tile([0,1,1],6)
        permutation,audit = f.permute_groups(np.column_stack([category,support]),rna)
        np.testing.assert_array_equal(support[permutation],support)
        np.testing.assert_array_equal(category[permutation],category)
        self.assertGreater(audit['identity_changed_fraction'],.9)

    def test_original_and_added_links_are_separate_and_missing_genes_do_not_match(self):
        dictionary = dict(genes=['a','b'],drug_names=['d0','d1'])
        original = sparse.csr_matrix([[1,0],[0,0]])
        added = sparse.csr_matrix([[0,0],[0,1]])
        sid,pairs,channels = f.target_links(np.array([0,0,1]),np.array(['a','b','missing']),
            np.array([0,1,2]),dictionary,[original,added])
        np.testing.assert_array_equal(sid,[0,1])
        np.testing.assert_array_equal(pairs,[0,1])
        np.testing.assert_array_equal(channels,[0,5])

    def test_pair_projection_fits_unique_covered_support_pairs_only(self):
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)/'association'
            directory.mkdir()
            np.savez(directory/'sites.npz',simple=rng.normal(size=(5,8)),names=np.arange(5),cids=np.arange(3))
            ids = np.array([0,1,3,5,8,11,14])
            payload = rng.normal(size=(len(ids),504)).astype(np.float32)
            np.savez(directory/'target.npz',ids=ids,payload=payload,gene_counts=np.arange(1,8),counts=np.ones((7,8)))
            enc = f.SiteFeatures(temp,'association')
            pairs = np.array([[0,0,1,1,1,2],[0,0,0,2,2,0]])
            table,state,_ = enc.transform('target',pairs)
            np.testing.assert_array_equal(state['fit_entity_ids'],[0,1*3,1*3+2])
            self.assertEqual(table.shape,(15,24))
            np.testing.assert_array_equal(table[2,8:],0)
            np.testing.assert_array_equal(table[ids,8],1)
            old = table.copy()
            enc.cache['target']['payload'][[1,4,5,6]] *= 100
            table,state2,_ = enc.transform('target',pairs)
            for key in state:
                np.testing.assert_array_equal(state[key],state2[key])
            np.testing.assert_array_equal(old[[0,3,5]],table[[0,3,5]])


if __name__ == '__main__':
    unittest.main()
