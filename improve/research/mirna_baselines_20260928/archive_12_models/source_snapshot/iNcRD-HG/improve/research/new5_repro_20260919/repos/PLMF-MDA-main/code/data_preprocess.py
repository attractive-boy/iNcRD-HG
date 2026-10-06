import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType



class Data_class(Dataset):

    def __init__(self, triple, drug_features, mirna_features):
        self.mirna_indices = triple[:, 0]
        self.drug_indices = triple[:, 1]
        self.label = triple[:, 2]
        self.drug_features = drug_features
        self.mirna_features = mirna_features

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        mirna_idx = self.mirna_indices[index]
        drug_idx = self.drug_indices[index]
        label = self.label[index]
        drug_smiles = self.drug_features['smiles'][drug_idx]
        drug_graph = self.drug_features['graph'][drug_idx]
        mirna_seq = self.mirna_features['sequence'][mirna_idx]
        mirna_encoded = self.mirna_features['encoded'][mirna_idx]

        return {
            'drug_smiles': drug_smiles,
            'drug_graph': drug_graph,
            'mirna_seq': mirna_seq,
            'mirna_encoded': mirna_encoded,
            'label': label,
            'drug_idx': drug_idx,
            'mirna_idx': mirna_idx
        }

def one_of_k_encoding_unk(value, choices):

    if value not in choices:
        value = choices[-1]
    return [int(value == item) for item in choices]

def smiles_to_graph(smiles, max_num_nodes = 100):

    atom_types = [
        'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
        'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
        'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
        'Pt', 'Hg', 'Pb', 'Unknown'
    ]
    hybrid_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        'other'
    ]
    degrees = [0, 1, 2, 3, 4, 5]
    # formal_charges = [-1, 0, 1]
    formal_charges = [-2, -1, 0, 1, 2]
    num_hs = [0, 1, 2, 3, 4]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():

        atom_type_enc = one_of_k_encoding_unk(atom.GetSymbol(), atom_types)

        hyb = atom.GetHybridization()
        hyb_val = hyb if hyb in hybrid_types[:-1] else 'other'
        hybrid_enc = one_of_k_encoding_unk(hyb_val, hybrid_types)

        degree_enc = one_of_k_encoding_unk(atom.GetDegree(), degrees)

        charge_enc = one_of_k_encoding_unk(atom.GetFormalCharge(), formal_charges)

        num_h_enc = one_of_k_encoding_unk(atom.GetTotalNumHs(), num_hs)

        aromatic = [int(atom.GetIsAromatic())]

        radical = [atom.GetNumRadicalElectrons()]

        in_ring = [int(atom.IsInRing())]

        features = (
            atom_type_enc
            + hybrid_enc
            + degree_enc
            + charge_enc
            + num_h_enc
            + aromatic
            + radical
            + in_ring
        )
        atom_features.append(features)
    # print("atom_features:", atom_features)

    num_atoms = len(atom_features)

    if num_atoms < max_num_nodes:

        padding = [[0] * len(atom_features[0]) for _ in range(max_num_nodes - num_atoms)]
        atom_features.extend(padding)
    else:

        atom_features = atom_features[:max_num_nodes]

    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])

    x = torch.tensor(np.array(atom_features), dtype=torch.float)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

def load_drug_data(file_path):
    """load drug data"""
    df = pd.read_excel(file_path)

    drug_features = {
        'smiles': {},
        'graph': {}
    }

    valid_drugs = []
    for idx, row in df.iterrows():
        smiles = row['SMILES']
        graph = smiles_to_graph(smiles)

        if graph is not None:
            drug_features['smiles'][idx] = smiles
            drug_features['graph'][idx] = graph
            valid_drugs.append(idx)

    print(f"loaded {len(valid_drugs)} drugs")

    return drug_features, valid_drugs


def encode_sequence(sequence, max_len=24):
    mapping = {'A': 1, 'U': 2, 'C': 3, 'G': 4}
    encoded = [mapping.get(base, 0) for base in sequence]
    # print("encoded:", encoded)

    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        encoded.extend([0] * (max_len - len(encoded)))

    return torch.tensor(encoded, dtype=torch.long)

def load_mirna_data(file_path):
    """load miRNA data"""
    df = pd.read_excel(file_path)

    mirna_features = {
        'sequence': {},
        'encoded': {}
    }

    valid_mirnas = []
    for idx, row in df.iterrows():
        sequence = row['miRNA_Sequence']

        if len(sequence) > 0:
            encoded_seq = encode_sequence(sequence)
            mirna_features['sequence'][idx] = sequence
            mirna_features['encoded'][idx] = encoded_seq
            valid_mirnas.append(idx)

    print(f"Successfully loaded {len(valid_mirnas)} miRNAs")
    # print("mirna_features:", mirna_features)
    return mirna_features, valid_mirnas

def load_data(args, test_ratio=0.2):
    """读取数据，转换为loader，返回特征和邻接关系"""
    print('Loading {0} seed{1} dataset...'.format(args.pos_sample, args.seed))

    # load drug and miRNA feature
    drug_features, valid_drugs = load_drug_data(args.drug_file)
    mirna_features, valid_mirnas = load_mirna_data(args.mirna_file)


    # load positive and negative samples
    positive = np.loadtxt(args.pos_sample, dtype=np.int64)
    print("positive:", positive.shape)

    # pos_sanple
    link_size = int(positive.shape[0])
    np.random.seed(args.seed)
    np.random.shuffle(positive)
    positive = positive[:link_size]

    # neg_sample
    negative_all = np.loadtxt(args.neg_sample, dtype=np.int64)
    np.random.shuffle(negative_all)
    negative = negative_all[:positive.shape[0]]
    print("positive examples: %d, negative examples: %d." % (positive.shape[0], negative.shape[0]))

    test_size = int(test_ratio * positive.shape[0])
    print("test_number:", test_size)

    positive = np.concatenate([positive, np.ones(positive.shape[0], dtype=np.int64).reshape(positive.shape[0], 1)],axis=1)
    negative = np.concatenate([negative, np.zeros(negative.shape[0], dtype=np.int64).reshape(negative.shape[0], 1)],axis=1)

    print("positive_negative: ", positive.shape, negative.shape)

    train_data = np.vstack((positive[:-test_size], negative[:-test_size]))
    test_data = np.vstack((positive[-test_size:], negative[-test_size:]))
    print("data: ", train_data.shape, test_data.shape)

    params = {'batch_size': args.batch, 'shuffle': True, 'num_workers': args.workers, 'drop_last': True}

    training_set = Data_class(train_data, drug_features, mirna_features)
    train_loader = DataLoader(training_set, **params)

    test_set = Data_class(test_data, drug_features, mirna_features)
    test_loader = DataLoader(test_set, **params)

    return train_loader, test_loader

def collate_fn(batch):
    drug_graphs = []
    drug_smiles = []
    mirna_seqs = []
    mirna_encoded = []
    labels = []
    drug_indices = []
    mirna_indices = []

    for item in batch:
        drug_graphs.append(item['drug_graph'])
        drug_smiles.append(item['drug_smiles'])
        mirna_seqs.append(item['mirna_seq'])
        mirna_encoded.append(item['mirna_encoded'])
        labels.append(item['label'])
        drug_indices.append(item['drug_idx'])
        mirna_indices.append(item['mirna_idx'])

    labels = torch.tensor(labels, dtype=torch.float)
    drug_indices = torch.tensor(drug_indices, dtype=torch.long)
    mirna_indices = torch.tensor(mirna_indices, dtype=torch.long)
    mirna_encoded = torch.stack(mirna_encoded, dim=0)

    return {
        'drug_graphs': drug_graphs,
        'drug_smiles': drug_smiles,
        'mirna_seqs': mirna_seqs,
        'mirna_encoded': mirna_encoded,
        'labels': labels,
        'drug_indices': drug_indices,
        'mirna_indices': mirna_indices

    }


