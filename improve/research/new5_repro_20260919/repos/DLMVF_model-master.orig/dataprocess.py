import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict

from fontTools.ttx import process
from rdkit import Chem
from torch_geometric.data import Data
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
seq_dict={}
seq_voc = "ACGU"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}

# 生成数据集的索引
def datasetindex():
    ass = pd.read_excel('data/miRNA_drug_matrix.xlsx', index_col=0)  # 关系矩阵导入
    po_rows, po_cols = np.where(ass.values == 1)
    ne_rows, ne_cols = np.where(ass.values != 1)
    Positive = json.load(open("data/Positive.txt"))
    Negetive = json.load(open("data/Negative.txt"))
    Potrain_fold = [[] for i in range(5)]
    Netrain_fold = [[] for i in range(5)]
    Povalid_fold = [[] for i in range(5)]
    Nevalid_fold = [[] for i in range(5)]
    Po_subset = [[] for i in range(5)]
    Ne_subset = [[] for i in range(5)]
    for i in range(5):
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
    opts = ['train', 'test']
    train_indices = {'rows': [], 'cols': []}
    test_indices = {'rows': [], 'cols': []}
    all_train_indices = []
    all_test_indices =[]

    for opt in opts:
        for i in range(5):
            if opt == 'train':
                train_indices['rows'] = np.concatenate((po_rows[Potrain_fold[i]], ne_rows[Netrain_fold[i]]))
                train_indices['cols'] = np.concatenate((po_cols[Potrain_fold[i]], ne_cols[Netrain_fold[i]]))
                all_train_indices.append(train_indices)
            elif opt == 'test':
                test_indices['rows'] = np.concatenate((po_rows[Povalid_fold[i]], ne_rows[Nevalid_fold[i]]))
                test_indices['cols'] = np.concatenate((po_cols[Povalid_fold[i]], ne_cols[Nevalid_fold[i]]))
                all_test_indices.append(test_indices)
    return all_train_indices,all_test_indices



def kfoldprepare():
    all_prots = []
    # 读取数据文件
    drugs = pd.read_excel('data/drug_id_smiles.xlsx')
    rna = pd.read_excel('data/miRNA_gen.xlsx')
    rna['Sequence'] = rna['Sequence'].apply(lambda x: x.strip().upper())

    gene = pd.read_csv('Predataprocess/miRNA_embeddings.txt', delimiter='\t', header=None)
    adj = pd.read_csv('data/MD_adjaceny.txt', delimiter=' ', header=None)
    ligands = drugs['smiles']
    proteins = rna['Sequence']

    ass = pd.read_excel('data/miRNA_drug_matrix.xlsx', index_col=0)  # 关系矩阵导入
    drugs = []
    prots = []

    all_train_indices,all_test_indices = datasetindex()
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)
    for t in proteins.keys():
        prots.append(proteins[t])

    affinity = np.asarray(ass)
    opts = ['train', 'test']
    all_lstrain = []
    all_lstest = []
    for opt in opts:
        for i in range(5):
            if opt == 'train':
                lstrain = []
                rows=all_train_indices[i]['rows']
                cols=all_train_indices[i]['cols']
                for pair_ind in range(len(rows)):
                    lstrain.append([drugs[cols[pair_ind]],
                                    prots[rows[pair_ind]],
                                    list(gene.iloc[rows[pair_ind], :]),
                                    affinity[rows[pair_ind], cols[pair_ind]]])

                all_lstrain.append(lstrain)

            elif opt == 'test':
                lstest = []
                rows=all_test_indices[i]['rows']
                cols=all_test_indices[i]['cols']
                for pair_ind in range(len(rows)):
                    lstest.append([drugs[cols[pair_ind]],
                               prots[rows[pair_ind]],
                               list(gene.iloc[rows[pair_ind], :]),
                               affinity[rows[pair_ind], cols[pair_ind]]])

                all_lstest.append(lstest)

    all_prots += list(set(prots))
    compound_iso_smiles=set(item[0] for item in all_lstrain[0]) | set(item[0] for item in all_lstest[0])

    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    adj = lalacians_norm(adj)
    edges_o = adj.nonzero()
    edge_index_o = torch.tensor(np.vstack((edges_o[0], edges_o[1])), dtype=torch.long)
    np.random.seed(0)
   
    num_nodes = max(adj.shape[0], adj.shape[1])
    features = np.random.normal(loc=0, scale=1, size=(num_nodes, 256))
    features_o = normalize(features)
    x_o = torch.tensor(features_o, dtype=torch.float)
    data_o = Data(x=x_o, edge_index=edge_index_o)
    train_data_list=[]
    test_data_list=[]
    for i in range(5):
        train_drugs, train_prots,train_gene,train_Y = [item[0] for item in all_lstrain[i]], [item[1] for item in all_lstrain[i]], [item[2] for item in all_lstrain[i]],[item[3] for item in all_lstrain[i]]
        XT = [seq_cat(t,24) for t in train_prots]
        train_sdrugs = [label_smiles(t, CHARISOSMISET, 100) for t in train_drugs]
        train_drugs, train_prots,train_gene,train_Y,train_seqdrugs = np.asarray(train_drugs), np.asarray(XT),np.asarray(train_gene),np.asarray(train_Y),np.asarray(train_sdrugs)

        test_drugs, test_prots,test_gene, test_Y = [item[0] for item in all_lstest[i]], [item[1] for item in all_lstest[i]], [item[2] for item in all_lstest[i]], [item[3] for item in all_lstest[i]]
        XT = [seq_cat(t,24) for t in test_prots]
        test_sdrugs=[label_smiles(t,CHARISOSMISET,100) for t in test_drugs]
        test_drugs, test_prots, test_gene, test_Y, test_seqdrugs = np.asarray(test_drugs), np.asarray(XT),np.asarray(test_gene),np.asarray(test_Y),np.asarray(test_sdrugs)

        # make data PyTorch Geometric ready
        print('preparing train.pt in pytorch format!')

        train_data = TestbedDataset(root='data/',dataset=None,
                                    xd=train_drugs, xt=train_prots, y=train_Y,xg=train_gene, z=train_seqdrugs,smile_graph=smile_graph,
                                    index=all_train_indices[i])
        print('preparing _test.pt in pytorch format!')
        test_data = TestbedDataset(root='data/',dataset=None,
                                    xd=test_drugs, xt=test_prots, y=test_Y, xg=test_gene, z=test_seqdrugs, smile_graph=smile_graph,
                                    index=all_test_indices[i])
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    return train_data_list, test_data_list,data_o


