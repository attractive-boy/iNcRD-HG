import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from utils import *


class Attention(nn.Module):
    # https://github.com/jindi-tju/U-GCN/blob/1bb9f95d2c4bfdb1e0a45de461baced95114d688/models.py#L47
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        # return (beta * z).sum(1), beta
        return beta


class MLP(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(MLP, self).__init__()

        self.T = nn.Parameter(torch.randn(in_size, hidden_size), requires_grad=True)

    def forward(self, z):
        z = z @ self.T
        return z


class MEGNN(nn.Module):
    def __init__(self, n, num_view, num_class, nfeats, nhid, threshold, max_iter, gamma, dropout=0.5):
        super(MEGNN, self).__init__()
        self.dropout = dropout
        self.MEGNN = torch.nn.ModuleList()
        self.Project = torch.nn.ModuleList()
        self.att = Attention(in_size=nhid)
        for dv in nfeats:
            self.MEGNN.append(MEGNN_m_iter(n, dv, nhid, threshold, max_iter, gamma))
            self.Project.append(MLP(dv, nhid))
        self.bn1 = nn.BatchNorm1d(nhid)
        self.B = nn.Parameter(1. / np.sqrt(nhid) * torch.randn(nhid, num_class), requires_grad=True)

    def reset_parameters(self):
        for i in range(len(self.MEGNN)):
            self.MEGNN[i].reset_parameters()
        # torch.nn.init.xavier_uniform_(self.B)

    def forward(self, X, adj):
        MEGNN_outputs = []
        for idx, model in enumerate(self.MEGNN):
            tmp_output, n_step = model(X[idx], adj[idx])
            # print(n_step)
            tmp_output = self.Project[idx](tmp_output)
            MEGNN_outputs.append(tmp_output)
        output = torch.stack(MEGNN_outputs, dim=1)
        att_vals = self.att(output)
        output = (output * att_vals).sum(1)
        # output = output.sum(1)
        output = self.bn1(output)
        output = F.dropout(output, self.dropout, training=self.training)
        output = output @ self.B
        return output
