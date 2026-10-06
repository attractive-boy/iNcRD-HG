import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
import scipy.sparse as sp

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset=None,
                 xd=None, xt=None, y=None, xg=None, z=None,
                 index=None,smile_graph=None,
                 transform=None, pre_transform=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)

        if xd is not None and xt is not None and y is not None and xg is not None and z is not None:

            self.data, self.slices = self.process(xd, xt, y, xg, z, smile_graph,index)
        else:
            raise ValueError("Input data cannot be None when loading data from memory.")

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

  
    def process(self, xd, xt, y, xg, z, smile_graph, index):
        count=0

        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)

        for i in range(data_len):
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            seqdrug=z[i]
            gene=xg[i]

            row_indices=index['rows']
            col_indices=index['cols']
            c_size, features, edge_index = smile_graph[smiles]
            if len(edge_index) == 0:
                count=count+1
                print(f'No edges for graph {i + 1}, skipping...',smiles)
                continue

            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.LongTensor([labels]))


            GCNData.target = torch.LongTensor([target])

            GCNData.gene= torch.LongTensor([gene])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # Add seqdrug as an attribute
            GCNData.__setitem__('seqdrug', torch.FloatTensor([seqdrug]))
            GCNData.__setitem__('row_indices', torch.LongTensor([row_indices[i]]))
            GCNData.__setitem__('col_indices', torch.LongTensor([col_indices[i]]))
            data_list.append(GCNData)
        print("去除不规则数量", count, "总数量为", data_len)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        # save preprocessed data:
        # torch.save((data, slices), self.processed_paths[0])
        return data, slices


def laplacian_norm(adj):
    adj += np.eye(adj.shape[0])   # add self-loop
    degree = np.array(adj.sum(1))
    D = []
    for i in range(len(degree)):
        if degree[i] != 0:
            de = np.power(degree[i], -0.5)
            D.append(de)
        else:
            D.append(0)
    degree = np.diag(np.array(D))
    norm_A = degree.dot(adj).dot(degree)

    return norm_A

def lalacians_norm(adj):
    row_degree = np.array(adj.sum(1))
    col_degree = np.array(adj.sum(0))
    
    D_row = []
    for i in range(len(row_degree)):
        if row_degree[i] != 0:
            de = np.power(row_degree[i], -0.5)
            D_row.append(de)
        else:
            D_row.append(0)
    D_row = np.diag(np.array(D_row))
    
    D_col = []
    for i in range(len(col_degree)):
        if col_degree[i] != 0:
            de = np.power(col_degree[i], -0.5)
            D_col.append(de)
        else:
            D_col.append(0)
    D_col = np.diag(np.array(D_col))
    norm_A = D_row.dot(adj).dot(D_col)
    return norm_A


def normalize(mx):
    '''Row-normalize sparse matrix'''
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx





