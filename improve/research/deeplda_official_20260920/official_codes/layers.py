import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GATLayer(nn.Module):
    """
    Simple GAT layer
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCNLayer(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, args):
        super(GCNLayer, self).__init__()
        self.args = args
        self.conv1_lnc_f = GCNConv(args.l_f_nfeat, args.nhid)
        self.act1_lnc_f = nn.Sequential(nn.ReLU(),
                              nn.Dropout(args.dropout))
        self.conv2_lnc_f = GCNConv(args.nhid, args.nhid)
        self.act2_lnc_f = nn.Sequential(nn.ReLU(),
                              nn.Dropout(args.dropout))
        self.conv3_lnc_f = GCNConv(args.nhid, args.nclass)
        
        self.conv1_gene_f = GCNConv(args.g_f_nfeat, args.nhid)
        self.act1_gene_f=nn.Sequential(nn.ReLU(),
                              nn.Dropout(args.dropout))
        self.conv2_gene_f = GCNConv(args.nhid, args.nhid)
        self.act2_gene_f = nn.Sequential(nn.ReLU(),
                              nn.Dropout(args.dropout))
        self.conv3_gene_f = GCNConv(args.nhid, args.nclass)
        
    def forward(self, dataset):
        x = self.act1_lnc_f(self.conv1_lnc_f(dataset['Lnc_f_features'], dataset['Lnc_f_edge_index']))
        x = self.act2_lnc_f(self.conv2_lnc_f(x, dataset['Lnc_f_edge_index']))
        x = self.conv3_lnc_f(x, dataset['Lnc_f_edge_index'])
        
        y = self.act1_gene_f(self.conv1_gene_f(dataset['Gene_f_features'], dataset['Gene_f_edge_index']))
        y = self.act2_gene_f(self.conv2_gene_f(y, dataset['Gene_f_edge_index']))
        y = self.conv3_gene_f(y, dataset['Gene_f_edge_index'])        
        #print(x.shape)
        #print(y.shape)
        #print(x2.shape)
        #print(y2.shape)
        return x, y

    
class CNNLayer(nn.Module):
    """
    Simple CNN layer
    """
    def __init__(self, args):
        super(CNNLayer, self).__init__()
        self.args = args
        self.cnn_x = nn.Conv1d(in_channels=args.l_f_nfeat,
                               out_channels=200,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.cnn_y = nn.Conv1d(in_channels=args.g_f_nfeat,
                               out_channels=200,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.globalAvgPool_x = nn.AvgPool2d((3, 3), (1, 1),padding=1)
        self.globalAvgPool_y = nn.AvgPool2d((3, 3), (1, 1),padding=1)
        
        self.fc_x = nn.Linear(in_features=200,
                             out_features=args.nclass)
        self.fc_y = nn.Linear(in_features=200,
                             out_features=args.nclass)

    def forward(self, x, x2):
        x=x.t()#print(x.shape)
        x = torch.relu(self.cnn_x(x.view(1, x.shape[0], x.shape[1])))#print(x.shape)
        x = torch.relu(torch.relu(self.globalAvgPool_x(x.view(1, x.shape[0], x.shape[1], x.shape[2]))))#print(x.shape)
        x = self.fc_x(x.view(x.shape[2], x.shape[3]).t())#print(x.shape)
   
        x2=x2.t()#print(x2.shape)
        x2 = torch.relu(self.cnn_y(x2.view(1, x2.shape[0], x2.shape[1])))#print(x2.shape)
        x2 = torch.relu(self.globalAvgPool_y(x2.view(1, x2.shape[0], x2.shape[1],x2.shape[2])))#print(x2.shape)
        x2 = self.fc_y(x2.view(x2.shape[2], x2.shape[3]).t())#print(x2.shape)
        return x, x2 
    
    
    
    
        