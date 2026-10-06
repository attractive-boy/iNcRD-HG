# %%

import pandas as pd
from torch_geometric.data import HeteroData
import torch
import torch_geometric.transforms as T
import numpy as np
import random

# %%
SEED = 666
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Extracting edges of homogeneous graphs from similarity matrix
def homo_sim_net(sim_file,sim_score):
    sim_mat = pd.read_csv(sim_file,header=None)
    sim_mat = np.abs(sim_mat)
    rows, columns = np.where(sim_mat > sim_score)
    sim_edge = np.vstack((rows, columns))
    print(sim_edge.shape)
    return sim_edge

# Extract numbers from column names
def extract_number(column_name):
    return int(column_name.split('-')[-1])

# %%

#####################################################################
# Save lncRNA information
def save_lncRNA_data():
    lncRNA_names =  pd.read_csv('./2.construct_network/lncRNA.csv')
    drug_names = pd.read_csv('./2.construct_network/Drug.csv').iloc[:, 1]
    lncRNADrug =  pd.read_csv('./2.construct_network/LncDrug_edge.csv')
    lncRNADrug = lncRNADrug.iloc[:, [0, 4]]
    lncRNA_mapping = {name: index for index, name in lncRNA_names['x'].items()}
    drug_mapping = {name: index for index, name in drug_names.items()}
    lncRNADrug['ncRNA_Name'] = lncRNADrug['ncRNA_Name'].map(lncRNA_mapping)
    lncRNADrug['CID'] = lncRNADrug['CID'].map(drug_mapping)
    lncRNADrug_edge = lncRNADrug.T
    lncRNA_edge_ken = homo_sim_net('./2.construct_network/LncSimMat_ken.csv',0.3)
    lncRNA_edge_pea = homo_sim_net('./2.construct_network/LncSimMat_pea.csv', 0.3)
    lncRNA_edge_spe = homo_sim_net('./2.construct_network/LncSimMat_spe.csv', 0.3)
    lncRNA_feature =  pd.read_csv('./2.construct_network/LncExp.csv', index_col=0).T

    lncRNA_info = {
        'lncRNA_feature': lncRNA_feature,
        'lncRNA_edge_ken': lncRNA_edge_ken,
        'lncRNA_edge_pea': lncRNA_edge_pea,
        'lncRNA_edge_spe':lncRNA_edge_spe,
        'lncRNADrug_edge': lncRNADrug_edge,
        'lncRNA_names': lncRNA_names
    }
    torch.save(lncRNA_info, './3.heter_data/lncRNA_info.pth')

# Save miRNA information
def save_miRNA_data():
    miRNA_names =  pd.read_csv('./2.construct_network/miRNA.csv')
    drug_names = pd.read_csv('./2.construct_network/Drug.csv').iloc[:, 1]
    miRNADrug =  pd.read_csv('./2.construct_network/MiDrug_edge.csv')
    miRNADrug = miRNADrug.iloc[:, [0, 4]]
    miRNA_mapping = {name: index for index, name in miRNA_names['x'].items()}
    drug_mapping = {name: index for index, name in drug_names.items()}
    miRNADrug['ncRNA_Name'] = miRNADrug['ncRNA_Name'].map(miRNA_mapping)
    miRNADrug['CID'] = miRNADrug['CID'].map(drug_mapping)
    miRNADrug_edge = miRNADrug.T
    miRNA_edge_ken = homo_sim_net('./2.construct_network/MiSimMat_ken.csv',0.3)
    miRNA_edge_pea = homo_sim_net('./2.construct_network/MiSimMat_pea.csv', 0.3)
    miRNA_edge_spe = homo_sim_net('./2.construct_network/MiSimMat_spe.csv', 0.3)
    miRNA_feature =  pd.read_csv('./2.construct_network/MiExp.csv', index_col=0).T
    sorted_columns = sorted(miRNA_feature.columns, key=extract_number)
    miRNA_feature = miRNA_feature[sorted_columns]

    miRNA_info = {
        'miRNA_feature': miRNA_feature,
        'miRNA_edge_ken': miRNA_edge_ken,
        'miRNA_edge_pea': miRNA_edge_pea,
        'miRNA_edge_spe':miRNA_edge_spe,
        'miRNADrug_edge': miRNADrug_edge,
        'miRNA_names': miRNA_names
    }
    torch.save(miRNA_info, './3.heter_data/miRNA_info.pth')

# Save drug information
def save_drug_data():
    drug_names = pd.read_csv('./2.construct_network/Drug.csv').iloc[:, 1]
    drug_feature = pd.read_csv('./2.construct_network/drug_features-chemBertA.csv', index_col=0)
    drug_edge =  homo_sim_net('./2.construct_network/drugSimMat.csv',0.3)
    drug_info = {
        'drug_names':drug_names,
        'drug_feature':drug_feature,
        'drug_edge':drug_edge
    }
    torch.save(drug_info,'./3.heter_data/drug_info.pth')

# Save lncRNA-miRNA information
def save_LncMiRNA_data():
    lncRNA_names = pd.read_csv('./2.construct_network/lncRNA.csv')
    miRNA_names = pd.read_csv('./2.construct_network/miRNA.csv')
    LncMiRNA = pd.read_csv('./2.construct_network/LncMiRNA_interaction.csv')
    lncRNA_mapping = {name: index for index, name in lncRNA_names['x'].items()}
    miRNA_mapping = {name: index for index, name in miRNA_names['x'].items()}
    LncMiRNA['lncRNA'] = LncMiRNA['lncRNA'].map(lncRNA_mapping)
    LncMiRNA['miRNA'] = LncMiRNA['miRNA'].map(miRNA_mapping)
    LncMi_edge = LncMiRNA.T

    LncMi_info = {
        'lncRNA_names':lncRNA_names,
        'miRNA_names':miRNA_names,
        'LncMi_edge':LncMi_edge
    }
    torch.save(LncMi_info,'./3.heter_data/LncMi_info.pth')


#####################################################################
def load_lncRNA_data(sim_method='ken'):
    lncRNA_info = torch.load('./3.heter_data/lncRNA_info.pth')
    lncRNADrug_edge = lncRNA_info['lncRNADrug_edge']
    if sim_method == 'ken':
      lncRNA_edge = lncRNA_info['lncRNA_edge_ken']
    if sim_method == 'pea':
      lncRNA_edge = lncRNA_info['lncRNA_edge_pea']
    if sim_method == 'spe':
      lncRNA_edge = lncRNA_info['lncRNA_edge_spe']
    lncRNA_feature = torch.tensor(lncRNA_info['lncRNA_feature'].values, dtype=torch.float)
    lncRNA_names = lncRNA_info['lncRNA_names']
    return lncRNADrug_edge, lncRNA_edge, lncRNA_feature,lncRNA_names

def load_miRNA_data(sim_method='ken'):
    miRNA_info = torch.load('./3.heter_data/miRNA_info.pth')
    miRNADrug_edge = miRNA_info['miRNADrug_edge']
    if sim_method == 'ken':
      miRNA_edge = miRNA_info['miRNA_edge_ken']
    if sim_method == 'pea':
      miRNA_edge = miRNA_info['miRNA_edge_pea']
    if sim_method == 'spe':
      miRNA_edge = miRNA_info['miRNA_edge_spe']
    miRNA_feature = torch.tensor(miRNA_info['miRNA_feature'].values, dtype=torch.float)
    miRNA_names = miRNA_info['miRNA_names']
    return miRNADrug_edge, miRNA_edge, miRNA_feature, miRNA_names

def load_drug_data():
    drug_info = torch.load('./3.heter_data/drug_info.pth')
    drug_names = drug_info['drug_names']
    drug_feature = torch.tensor(drug_info['drug_feature'].values, dtype=torch.float)
    drug_edge = drug_info['drug_edge']
    return drug_names, drug_feature, drug_edge

def load_LncMiRNA_data():
    lncRNA_miRNA_info = torch.load('./3.heter_data/LncMi_info.pth')
    LncMi_edge = lncRNA_miRNA_info['LncMi_edge']
    return LncMi_edge

############################################################################
# Construct heterogeneous network
def construct_heterogeneous_network(lncRNADrug_edge, lncRNA_edge, lncRNA_feature,lncRNA_names,miRNADrug_edge, miRNA_edge, miRNA_feature, miRNA_names,drug_names, drug_feature, drug_edge,LncMi_edge):
    data = HeteroData()
    data['lncRNA'].node_id = torch.arange(len(lncRNA_names))
    data['miRNA'].node_id = torch.arange(len(miRNA_names))
    data['drug'].node_id = torch.arange(len(drug_names))

    data['lncRNA'].x = (lncRNA_feature - torch.min(lncRNA_feature, dim=1, keepdim=True).values) / (torch.max(lncRNA_feature, dim=1, keepdim=True).values - torch.min(lncRNA_feature, dim=1, keepdim=True).values + 1e-8)
    data['miRNA'].x = (miRNA_feature - torch.min(miRNA_feature, dim=1, keepdim=True).values) / (torch.max(miRNA_feature, dim=1, keepdim=True).values - torch.min(miRNA_feature, dim=1, keepdim=True).values)
    data['drug'].x = (drug_feature - torch.min(drug_feature, dim=1, keepdim=True).values) / (torch.max(drug_feature, dim=1, keepdim=True).values - torch.min(drug_feature, dim=1, keepdim=True).values)

    data['lncRNA', 'LncLnc', 'lncRNA'].edge_index = torch.tensor(lncRNA_edge, dtype=torch.long)
    #print("lncRNADrug_edge type:", type(lncRNADrug_edge.values))
    data['lncRNA', 'LncDrug', 'drug'].edge_index = torch.tensor(lncRNADrug_edge.values, dtype=torch.long)
    data['miRNA', 'MiMi', 'miRNA'].edge_index = torch.tensor(miRNA_edge, dtype=torch.long)
    data['miRNA', 'MiDrug', 'drug'].edge_index = torch.tensor(miRNADrug_edge.values, dtype=torch.long)
    data['drug', 'DrugDrug', 'drug'].edge_index = torch.tensor(drug_edge, dtype=torch.long)
    data['lncRNA', 'LncMi', 'miRNA'].edge_index = torch.tensor(LncMi_edge.values, dtype=torch.long)
    return data
# %%
def check_intersect(edge_type, RNA_type):
    train_data = torch.load('./3.heter_data/' + edge_type + '_train_data.pth')
    val_data = torch.load('./3.heter_data/' + edge_type + '_val_data.pth')
    test_data = torch.load('./3.heter_data/' + edge_type + '_test_data.pth')
        # %%
    train_edge_index = train_data[RNA_type, edge_type, 'drug'].edge_index
    val_edge_label_index = test_data[RNA_type, edge_type, 'drug'].edge_label_index
    indices = torch.where(test_data[RNA_type, edge_type, 'drug'].edge_label == 1)[0]
    val_edge_label_index_new = val_edge_label_index[:, indices]
    train_edges = torch.stack((train_edge_index[0], train_edge_index[1]), dim=1)
    val_edges = torch.stack((val_edge_label_index_new[0], val_edge_label_index_new[1]), dim=1)
    train_edges_set = set(map(tuple, train_edges.tolist()))
    val_edges_set = set(map(tuple, val_edges.tolist()))
    intersection = train_edges_set.intersection(val_edges_set)
    print('intersection: ',intersection)
# %%

if __name__ == '__main__':
    # %%
    device = torch.device('cuda')

    # save data
    save_lncRNA_data()
    save_miRNA_data()
    save_drug_data()
    save_LncMiRNA_data()

    # load data
    lncRNADrug_edge, lncRNA_edge, lncRNA_feature,lncRNA_names = load_lncRNA_data('ken')
    miRNADrug_edge, miRNA_edge, miRNA_feature, miRNA_names = load_miRNA_data('ken')
    drug_names, drug_feature, drug_edge = load_drug_data()
    LncMi_edge = load_LncMiRNA_data()
    print("lncRNADrug_edge shape:", lncRNADrug_edge.shape)
    # %%
    data = construct_heterogeneous_network(lncRNADrug_edge, lncRNA_edge, lncRNA_feature,lncRNA_names,miRNADrug_edge, miRNA_edge, miRNA_feature, miRNA_names,drug_names, drug_feature, drug_edge,LncMi_edge)
    data = T.ToUndirected()(data)
    data = T.AddSelfLoops()(data)
    # %%
    data.to(device=device)

    # %%
    RNA_type_all = [['lncRNA', 'LncDrug'], ['miRNA', 'MiDrug']]

    for RNA_type, edge_type in RNA_type_all:
        print(RNA_type)
        print(edge_type)

        # %%
        transform = T.RandomLinkSplit(
            num_val=0.2,
            num_test=0.2,
            neg_sampling_ratio=2.0,
            is_undirected=True,
            add_negative_train_samples=True,
            edge_types=(RNA_type, edge_type, 'drug'),
            rev_edge_types=('drug', 'rev_' + edge_type, RNA_type),
        )
        train_data, val_data, test_data = transform(data)
        train_file = './3.heter_data/' + edge_type + '_train_data.pth'
        val_file = './3.heter_data/' + edge_type + '_val_data.pth'
        test_file = './3.heter_data/' + edge_type + '_test_data.pth'
        torch.save(train_data,train_file)
        torch.save(val_data,val_file)
        torch.save(test_data, test_file)
        check_intersect(edge_type, RNA_type)
# %%
