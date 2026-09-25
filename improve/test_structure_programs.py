import unittest

import numpy as np
import torch
from scipy import sparse

from improve.test_graph_tabm import graph_fixture
from improve.structure_program_modules import (
    Program, ProgramEncoder, degree_swaps, grammar, nonbacktracking_five,
    normalize, select_programs,
)


class StructureProgramsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_grammar_types_and_known_counts_are_excluded(self):
        with self.assertRaises(ValueError):
            Program(('A','B'))
        with self.assertRaises(ValueError):
            Program(('C','T'))
        programs = grammar()
        self.assertEqual(len(programs),len({x.name for x in programs}))
        self.assertNotIn('count:S.A',[x.name for x in programs])
        self.assertIn('nonbacktracking:A.a.A.a.A',[x.name for x in programs])

    def test_nonbacktracking_counts_match_exhaustive_walk_enumeration(self):
        a = np.array([[1,1,0],[0,1,1],[1,0,1],[1,0,0]],dtype=float)
        adj = np.block([[np.zeros((4,4)),a],[a.T,np.zeros((3,3))]])
        expected = np.zeros_like(a)

        def walk(source,node,previous,depth):
            if depth == 5:
                if node >= 4:
                    expected[source,node-4] += 1
                return
            for nxt in np.flatnonzero(adj[node]):
                if nxt != previous:
                    walk(source,int(nxt),node,depth+1)

        for source in range(4):
            walk(source,source,-1,0)
        np.testing.assert_array_equal(nonbacktracking_five(a),expected)

    def test_normalized_typed_walk_matches_independent_dense_product(self):
        g,_,_ = graph_fixture()
        encoder = ProgramEncoder(g,'LncDrug')
        support = np.array([np.arange(8),np.arange(8)%6])
        pairs = np.array([[0,1,2],[3,4,5]])
        programs = [Program(('C','T','B'),mode) for mode in ('count','walk','symmetric')]
        result = encoder.evaluate(pairs,support,programs)
        for j,mode in enumerate(('count','walk','symmetric')):
            matrices = []
            for symbol in ('C','T','B'):
                x = encoder.fixed[symbol].toarray()
                if mode == 'walk':
                    x = x / np.maximum(x.sum(1,keepdims=True),1)
                elif mode == 'symmetric':
                    x = x / np.sqrt(np.maximum(x.sum(1,keepdims=True),1)*np.maximum(x.sum(0,keepdims=True),1))
                matrices.append(x)
            expected = matrices[0] @ matrices[1] @ matrices[2]
            np.testing.assert_allclose(result[:,j],expected[pairs[0],pairs[1]],atol=1e-7)

    def test_support_positive_is_rejected_but_graph_target_storage_is_ignored(self):
        g,_,_ = graph_fixture()
        support = np.array([[0,1,2],[0,1,2]])
        pairs = np.array([[0,1],[3,4]])
        programs = grammar(3)
        with self.assertRaises(ValueError):
            ProgramEncoder(g,'LncDrug').evaluate(support,support,programs)
        original = ProgramEncoder(g,'LncDrug').evaluate(pairs,support,programs)
        g['lncRNA','LncDrug','drug'].edge_index = torch.from_numpy(np.concatenate([support,pairs],1))
        g['drug','rev_LncDrug','lncRNA'].edge_index = g['lncRNA','LncDrug','drug'].edge_index.flip(0)
        changed = ProgramEncoder(g,'LncDrug').evaluate(pairs,support,programs)
        np.testing.assert_array_equal(original,changed)

    def test_double_edge_swaps_preserve_degrees_and_forbidden_queries(self):
        support = np.array([np.arange(12),np.arange(12)%6])
        query = np.array([np.arange(12),(np.arange(12)+2)%6])
        changed,audit = degree_swaps(support,(12,6),query,42,5)
        for axis,n in [(0,12),(1,6)]:
            np.testing.assert_array_equal(np.bincount(support[axis],minlength=n),np.bincount(changed[axis],minlength=n))
        self.assertFalse(set(map(tuple,changed.T)) & set(map(tuple,query.T)))
        self.assertTrue(audit['completed'])
        self.assertGreater(audit['fraction_edges_changed'],0)
        repeat,_ = degree_swaps(support,(12,6),query,42,5)
        np.testing.assert_array_equal(repeat,changed)

    def test_null_degeneracy_has_explicit_finite_raw_fallback(self):
        g,pairs,y = graph_fixture()
        query = np.array([[0,1,2],[3,4,5]])
        encoder = ProgramEncoder(g,'LncDrug')
        programs = [Program(('C','B'))]
        raw,corrected,audit = encoder.with_null(query,pairs[:,y==1],programs,2,12,2)
        self.assertTrue(np.isfinite(corrected).all())
        np.testing.assert_array_equal(raw,corrected)
        self.assertEqual(audit['fallback_fraction'],1.)
        self.assertFalse(audit['all_swaps_completed'])

    def test_node_renumbering_invariance_with_coupled_null_draws(self):
        g,_,_ = graph_fixture()
        support = np.array([np.arange(8),np.arange(8)%6])
        query = np.array([[0,1,2],[3,4,5]])
        programs = [Program(('S','A'),'walk'),Program(('C','B','b','T','B'),'symmetric')]
        before = ProgramEncoder(g,'LncDrug').with_null(query,support,programs,2,12,2)
        rng = np.random.RandomState(91)
        perm = {nt:rng.permutation(g[nt].num_nodes) for nt in g.node_types}
        changed = g.clone()
        for nt in g.node_types:
            changed[nt].x = g[nt].x[np.argsort(perm[nt])]
        for key,e in g.edge_index_dict.items():
            changed[key].edge_index = torch.from_numpy(np.stack([perm[key[0]][e[0]],perm[key[2]][e[1]]]))
        map_pairs = lambda x: np.stack([perm['lncRNA'][x[0]],perm['drug'][x[1]]])
        after = ProgramEncoder(changed,'LncDrug').with_null(map_pairs(query),map_pairs(support),programs,2,12,2)
        np.testing.assert_allclose(before[0],after[0],rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(before[1],after[1],rtol=1e-6,atol=1e-6)

    def test_training_only_selection_finds_signal_with_fixed_budget(self):
        rng = np.random.RandomState(4)
        y = rng.randint(0,2,size=300).astype(float)
        programs = grammar()[:24]
        x = rng.exponential(size=(300,len(programs)))
        x[:,0] += 20*y
        base = rng.exponential(size=(300,36))
        a = select_programs(base,x,y,programs,k=5)
        b = select_programs(base,x,y,programs,k=5)
        self.assertEqual(a['selected'][0],0)
        self.assertEqual(a,b)
        self.assertEqual(len(a['selected']),len(a['random']))
        self.assertEqual(len(set(a['selected'])),5)


if __name__ == '__main__':
    unittest.main()
