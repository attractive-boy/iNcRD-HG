import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        # print(line[0][0])
        # print(i,"========", ch)
        X[i] = smi_ch_ind[ch]
    return X

def seq_cat(prot,max_seq_len):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64, ">": 65, "<": 66}

CHARISOSMILEN = 66


all_prots = []
fpath='data/'
drugs = pd.read_excel('data/drug_id_smiles.xlsx')
rna = pd.read_excel('data/miRNA_sequences.xlsx')
ligands = drugs['smiles']
proteins = rna['Sequence']

ass=pd.read_excel('data/miRNA_drug_matrix.xlsx', index_col=0)   
Positive = json.load(open("data/Positive.txt"))
Negetive = json.load(open("data/Negative.txt"))
Potrain_fold=[[] for i in range(5)]
Netrain_fold=[[] for i in range(5)]
Povalid_fold=[[] for i in range(5)]
Nevalid_fold=[[] for i in range(5)]
Po_subset = [[] for i in range(5)]
Ne_subset=[[] for i in range(5)]
for i in  range(5):
    Po_subset[i] = [ee for ee in Positive[i]]
    Ne_subset[i] = [ee for ee in Negetive[i]]
for i in range(5):
    for j in range(5):
        if i == j:
            continue
        Potrain_fold[i] += Po_subset[j]
        Netrain_fold[i] += Ne_subset[j]
    Povalid_fold[i] = Po_subset[i]
    Nevalid_fold[i] = Ne_subset[i]

drugs = []
prots = []
for d in ligands.keys():
    lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
    drugs.append(lg)
for t in proteins.keys():
    prots.append(proteins[t])

affinity = np.asarray(ass)

opts = ['train', 'test']

for opt in opts:
    po_rows, po_cols = np.where(ass.values == 1)
    ne_rows, ne_cols = np.where(ass.values != 1)
    for i in range(5):
        if opt == 'train':
            rows = np.concatenate((po_rows[Potrain_fold[i]], ne_rows[Netrain_fold[i]]))
            cols = np.concatenate((po_cols[Potrain_fold[i]], ne_cols[Netrain_fold[i]]))
        elif opt == 'test':
            rows = np.concatenate((po_rows[Povalid_fold[i]], ne_rows[Nevalid_fold[i]]))
            cols = np.concatenate((po_cols[Povalid_fold[i]], ne_cols[Nevalid_fold[i]]))

        with open('data/processed/'+ opt +str(i)+ '.csv', 'w') as f:
            f.write('compound_iso_smiles,target_sequence,affinity\n')
            for pair_ind in range(len(rows)):
                ls = []
                ls += [drugs[cols[pair_ind]]]
                ls += [prots[rows[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                f.write(','.join(map(str, ls)) + '\n')


all_prots += list(set(prots))
seq_voc = "ACGU"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)

compound_iso_smiles = []


opts = ['train', 'test']

for opt in opts:
    df = pd.read_csv('data/processed/' + opt +str(i)+ '.csv')
    compound_iso_smiles += list(df['compound_iso_smiles'])
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g

for i in range(5):
    processed_data_file_train = 'data/processed/'  + '_train'+str(i)+'.pt'
    processed_data_file_test = 'data/processed/'+ '_test'+str(i)+'.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        df = pd.read_csv('data/processed/train' + str(i)+ '.csv')
        train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
            df['affinity'])
        XT = [seq_cat(t,24) for t in train_prots]
        train_sdrugs = [label_smiles(t, CHARISOSMISET, 100) for t in train_drugs]
        train_drugs, train_prots, train_Y,train_seqdrugs = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y),np.asarray(train_sdrugs)
        df = pd.read_csv('data/processed/test'+str(i)+ '.csv')
        test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
            df['affinity'])
        XT = [seq_cat(t,24) for t in test_prots]
        test_sdrugs=[label_smiles(t,CHARISOSMISET,100) for t in test_drugs]
        test_drugs, test_prots, test_Y,test_seqdrugs = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y),np.asarray(test_sdrugs)


        print('preparing NO:', i,'train.pt in pytorch format!')

        train_data = TestbedDataset(root='data/',dataset='train'+str(i), xd=train_drugs, xt=train_prots, y=train_Y, z=train_seqdrugs,
                                    smile_graph=smile_graph)
        print('preparing ',   '_test.pt in pytorch format!')
        test_data = TestbedDataset(root='data/',dataset='test'+str(i),  xd=test_drugs, xt=test_prots, y=test_Y, z=test_seqdrugs,
                                   smile_graph=smile_graph)
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')
    else:
        print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')
