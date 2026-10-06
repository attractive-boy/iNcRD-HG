import sys

import numpy as np
import scipy.sparse as sp
import torch
# import torch_sparse
import pickle as pkl
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn import metrics, manifold


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def f1_test(output, labels):
    preds = output.max(1)[1].type_as(labels)
    # correct = preds.eq(labels).double()
    f1 = f1_score(preds.detach().cpu().numpy(), labels.detach().cpu().numpy(), average='macro')
    return f1


def sparse_mx_to_torch_sparse_tensor(sparse_mx, device=None):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    tensor = torch.sparse.FloatTensor(indices, values, shape)
    if device is not None:
        tensor = tensor.to(device)
    return tensor

def get_normalized_adj(adj):
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def draw_plt(output_, labels, datasets):
    output_ = output_.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    X_tsne = manifold.TSNE(n_components=2, learning_rate=100, random_state=42).fit_transform(output_)
    plt.figure(figsize=(8, 6))
    # plt.title('Dataset : ' + dataset_name + '   (Label rate : 20 nodes per class)')

    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=8, cmap='rainbow')
    # scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=8)
    handles, _ = scatter.legend_elements(prop='colors')
    labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    plt.legend(handles, labels, loc='upper right')
    # plt.colorbar(ticks=range(5))
    plt.savefig('./results/tsne/' + datasets + '64.svg')
    plt.show()
