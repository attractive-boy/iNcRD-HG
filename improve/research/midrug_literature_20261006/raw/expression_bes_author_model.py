import os
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
from torch import tensor
import torch.nn.functional as F
from utils import load_conv_weights_path
from torch.nn import Linear
from torch_geometric.nn import GraphConv, GCNConv, ChebConv, GATConv, SGConv, SSGConv, SAGEConv, GATv2Conv

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size,seq_len, embed_dim = x.size()
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        z = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(z, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attended_values = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.fc(attended_values) + x
        return output
class model_repulsive(nn.Module):
    def __init__(self, embed_dim, num_heads,dataset,dataset_name,seed):
        super().__init__()
        self.layer = Layer(dataset,dataset_name,seed)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
    def forward(self, data):
        hidden_1, enc1, enc2 = self.layer(data)
        hidden_11 = self.attention(hidden_1)  
        hidden_111 ,_,_= self.layer(data)
        hidden_2 = hidden_11 + hidden_111
        hidden_3 = self.attention(hidden_2)  
        hidden_3 = hidden_3[:, 0, :]
        return hidden_3

class Layer(nn.Module):
    def __init__(self,dataset,dataset_name,seed):
        super(Layer, self).__init__()
        path1,path2 = load_conv_weights_path(dataset_name,seed)
        if dataset_name == 'WikiCS':
            self.enc_1 = Net_ChebConv(dataset)
            self.enc_2 = Net_GATv2Conv(dataset)
            self.enc_1.load_state_dict(torch.load(path1), strict=False)
            self.enc_2.load_state_dict(torch.load(path2), strict=False)
        elif dataset_name =="Cora":
            self.enc_1 = Net_ChebConv(dataset)
            self.enc_2 = Net_GCNConv(dataset)
            self.enc_1.load_state_dict(torch.load(path1), strict=False)
            self.enc_2.load_state_dict(torch.load(path2), strict=False)
        elif dataset_name =="CiteSeer":
            self.enc_1 = Net_SGConv(dataset)
            self.enc_2 = Net_SSGConv(dataset)
            self.enc_1.load_state_dict(torch.load(path1), strict=False)
            self.enc_2.load_state_dict(torch.load(path2), strict=False)
        elif dataset_name =="PubMed":
            self.enc_1 = Net_ChebConv(dataset)
            self.enc_2 = Net_GraphConv(dataset)
            self.enc_1.load_state_dict(torch.load(path1), strict=False)
            self.enc_2.load_state_dict(torch.load(path2), strict=False)
        for param in self.enc_1.parameters():
            param.requires_grad = False
        for param in self.enc_2.parameters():
            param.requires_grad = False
    def forward(self, data):
        enc_1 = self.enc_1(data)
        enc_2 = self.enc_2(data)
        enc = torch.cat((enc_1, enc_2), dim=1)
        enc = enc.unsqueeze(1).expand(-1,1, -1)
        return enc,enc_1,enc_2
class Net_GraphConv(torch.nn.Module):
    def __init__(self, dataset):
        super(Net_GraphConv, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, 16)
    def reset_parameters(self):
        self.conv1.reset_parameters()
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return x
class Net_GCNConv(torch.nn.Module):
    def __init__(self, dataset):
        super(Net_GCNConv, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
    def reset_parameters(self):
        self.conv1.reset_parameters()
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return x
class Net_ChebConv(torch.nn.Module):
    def __init__(self, dataset):
        super(Net_ChebConv, self).__init__()
        self.conv1 = ChebConv(dataset.num_features, 16,K=3)
    def reset_parameters(self):
        self.conv1.reset_parameters()
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return x
class Net_GATConv(torch.nn.Module):
    def __init__(self, dataset):
        super(Net_GATConv, self).__init__()
        self.conv1 = GATConv(dataset.num_features, 16)
    def reset_parameters(self):
        self.conv1.reset_parameters()
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return x    
class Net_GATv2Conv(torch.nn.Module):
    def __init__(self, dataset):
        super(Net_GATv2Conv, self).__init__()
        self.conv1 = GATv2Conv(dataset.num_features, 16)
    def reset_parameters(self):
        self.conv1.reset_parameters()
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return x 
class Net_SAGEConv(torch.nn.Module):
    def __init__(self, dataset):
        super(Net_SAGEConv, self).__init__()
        self.conv1 = SAGEConv(dataset.num_features, 16)
    def reset_parameters(self):
        self.conv1.reset_parameters()
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return x    
class Net_SGConv(torch.nn.Module):
    def __init__(self, dataset):
        super(Net_SGConv, self).__init__()
        self.conv1 = SGConv(dataset.num_features, 16)
    def reset_parameters(self):
        self.conv1.reset_parameters()
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return x    
class Net_SSGConv(torch.nn.Module):
    def __init__(self, dataset):
        super(Net_SSGConv, self).__init__()
        self.conv1 = SSGConv(dataset.num_features, 16,0.1)
    def reset_parameters(self):
        self.conv1.reset_parameters()
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return x
class model_gravite(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(model_gravite, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
    def forward(self, x):
        hidden_1 =x
        hidden_1 = hidden_1.unsqueeze(1).expand(-1,1, -1)
        hidden_2 = self.attention(hidden_1)
        hidden_2 = hidden_1 + hidden_2
        hidden_2 = self.attention(hidden_2)
        hidden_2 = hidden_2[:,0,:]
        return hidden_2
class classificationGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,dataset_name):
        super(classificationGCN, self).__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K=3)
        self.conv2 = ChebConv(hidden_channels, out_channels, K=3)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x1 = self.conv2(x, edge_index)
        x = self.bn2(x1)
        x = F.relu(x)  
        x = self.dropout(x)
        log_probs = F.log_softmax(x, dim=1)
        return log_probs,x1

class classificationMLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(classificationMLP, self).__init__()
        self.conv1 = Linear(in_channels, hidden_channels)
        self.conv2 = Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x1 = x
        x = self.bn2(x)
        x = F.relu(x)  
        x = self.dropout(x)
        log_probs = F.log_softmax(x, dim=1)
        return log_probs,x1