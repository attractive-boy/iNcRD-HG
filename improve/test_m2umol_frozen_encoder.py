"""Run with .venv-m2umol; exercise the actual released checkpoint."""
from pathlib import Path
import unittest

import torch
from rdkit import Chem
from torch_geometric.data import Batch

from improve.m2umol_frozen_encoder import M2UMolFrozen2D,graph_builder


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'.external/M2UMol_20260914'
CHECKPOINT = ROOT/'data_external/m2umol_20260914/pre-trained_M2UMol.pt'


class PretrainedEncoderTests(unittest.TestCase):
    def test_missing_input_weights_are_rejected_and_initialization_cannot_change_output(self):
        torch.set_num_threads(4)
        state = torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
        torch.manual_seed(1)
        first = M2UMolFrozen2D(SOURCE)
        first.load_pretrained(state)
        torch.manual_seed(991)
        second = M2UMolFrozen2D(SOURCE)
        second.load_pretrained(state)
        for key,value in first.state_dict().items():
            torch.testing.assert_close(value,second.state_dict()[key],rtol=0,atol=0)
        create = graph_builder(SOURCE)
        smiles = ['CCO','c1ccccc1','N.N.Cl[Pt+2]Cl']
        graphs = [create(Chem.MolFromSmiles(s),s) for s in smiles]
        with torch.inference_mode():
            a = first(Batch.from_data_list(graphs))
            b = second(Batch.from_data_list(graphs))
            reverse = second(Batch.from_data_list(graphs[::-1])).flip(0)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(a,reverse,rtol=1e-5,atol=1e-4)
        reduced = {key:value for key,value in state.items() if key!='linx.atom_embedding_list.0.weight'}
        with self.assertRaisesRegex(ValueError,'Missing pretrained 2D state'):
            second.load_pretrained(reduced)

    def test_no_salt_stripping_and_edgeless_inputs_are_explicitly_rejected(self):
        create = graph_builder(SOURCE)
        smiles = 'CCO.[Na+]'
        mol = Chem.MolFromSmiles(smiles)
        graph = create(mol,smiles)
        self.assertEqual(graph.num_nodes,4)
        self.assertEqual(tuple(graph.x.shape),(4,9))
        with self.assertRaisesRegex(ValueError,'edgeless'):
            create(Chem.MolFromSmiles('[Pt+2]'),'[Pt+2]')


if __name__=='__main__':
    unittest.main()
