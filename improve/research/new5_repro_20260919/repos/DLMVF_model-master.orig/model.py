import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp


class Attention(nn.Module):
    def __init__(self, feat_in, hidden=16):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(feat_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1, bias=False)
        )

    def forward(self, x):
        scores = self.project(x)               # (B, N, 1)
        weights = torch.softmax(scores, dim=1) # (B, N, 1)
        out = (weights * x).sum(dim=1)         # (B, F)
        return out, weights.squeeze(-1)


class DenseEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class Conv1DMultiScale(nn.Module):
    def __init__(self, embed_dim, n_filters, out_dim, dropout=0.2):
        super().__init__()
        self.dropout_rate = dropout

        def make_path(kernel_size):
            padding = kernel_size // 2
            return nn.Sequential(
                nn.Conv1d(embed_dim, n_filters * 2, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(n_filters * 2, n_filters * 4, kernel_size=kernel_size, padding=padding),
                nn.ReLU()
            )

        self.path_k4 = make_path(4)
        self.path_k3 = make_path(3)
        self.path_k2 = make_path(2)

        self.fc_proj = nn.Linear(n_filters * 4, out_dim)

    def forward(self, embedded_seq):
        p1 = self.path_k4(embedded_seq)
        p1 = F.max_pool1d(p1, p1.size(2)).squeeze(2)
        p1 = self.fc_proj(p1)

        p2 = self.path_k3(embedded_seq)
        p2 = F.max_pool1d(p2, p2.size(2)).squeeze(2)
        p2 = self.fc_proj(p2)

        p3 = self.path_k2(embedded_seq)
        p3 = F.max_pool1d(p3, p3.size(2)).squeeze(2)
        p3 = self.fc_proj(p3)

        return torch.cat([p1, p2, p3], dim=1)


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, proj_dim, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, 512)
        self.fc2 = nn.Linear(512, proj_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MDGraphEncoder(nn.Module):
    def __init__(self, in_dim=256, mid_dim=128, out_dim=64, fc_mid=128, proj_dim=256, dropout=0.2):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, mid_dim)
        self.gcn2 = GCNConv(mid_dim, out_dim)
        self.fc1 = nn.Linear(out_dim, fc_mid)
        self.fc2 = nn.Linear(fc_mid, proj_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index)
        h = self.relu(h)
        h = self.gcn2(h, edge_index)
        h = self.relu(h)
        h = self.fc1(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return h


class DLMVF(nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128, num_smiles_chars=66,
                 num_drug_node_feats=78, num_mirna_chars=25, proj_dim=256, dropout=0.2, drug_offset=1563):
        super().__init__()
        self.proj_dim = proj_dim
        self.drug_offset = drug_offset
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.smiles_embedding = nn.Embedding(num_smiles_chars + 1, embed_dim)
        self.smiles_encoder = Conv1DMultiScale(embed_dim, n_filters, proj_dim, dropout)
        self.smiles_reduce = nn.Conv1d(proj_dim * 3, proj_dim, kernel_size=1)

        self.drug_graph_encoder = GCNEncoder(num_drug_node_feats, num_drug_node_feats * 2, proj_dim, dropout)

        self.gene_encoder = DenseEncoder(in_dim=512, out_dim=proj_dim, hidden=256, dropout=dropout)

        self.mirna_embedding = nn.Embedding(num_mirna_chars + 1, embed_dim)
        self.mirna_encoder = Conv1DMultiScale(embed_dim, n_filters, proj_dim, dropout)
        self.mirna_reduce = nn.Conv1d(proj_dim * 3, proj_dim, kernel_size=1)

        self.md_graph_encoder = MDGraphEncoder(in_dim=256, proj_dim=proj_dim, dropout=dropout)

        self.attention = Attention(proj_dim)
        self.classifier_fc = nn.Sequential(
            nn.Linear(proj_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_output),
            nn.Sigmoid()
        )

    def forward(self, drug_graph_data, bipartite_graph_data):
        drug_graph_feat = self.drug_graph_encoder(drug_graph_data.x, drug_graph_data.edge_index, drug_graph_data.batch)

        smiles_seq = drug_graph_data.seqdrug.long()
        smiles_embedding = self.smiles_embedding(smiles_seq).permute(0, 2, 1)
        smiles_multi_feat = self.smiles_encoder(smiles_embedding)
        smiles_seq_feat = self.smiles_reduce(smiles_multi_feat.unsqueeze(2)).squeeze(2)

        gene_feat = self.gene_encoder(drug_graph_data.gene.float())

        mirna_seq = drug_graph_data.target.long()
        mirna_embedding = self.mirna_embedding(mirna_seq).permute(0, 2, 1)
        mirna_multi_feat = self.mirna_encoder(mirna_embedding)
        mirna_seq_feat = self.mirna_reduce(mirna_multi_feat.unsqueeze(2)).squeeze(2)

     
        md_node_feats = self.md_graph_encoder(bipartite_graph_data.x, bipartite_graph_data.edge_index)
        rna_md_feat = md_node_feats[drug_graph_data.row_indices, :]
        drug_md_feat = md_node_feats[drug_graph_data.col_indices + self.drug_offset, :]

      
        drug_feature_stack = torch.stack([smiles_seq_feat, drug_graph_feat, drug_md_feat], dim=1)
        drug_emb, _ = self.attention(drug_feature_stack)
        drug_emb = self.dropout(drug_emb)

        mirna_feature_stack = torch.stack([gene_feat, rna_md_feat, mirna_seq_feat], dim=1)
        mirna_emb, _ = self.attention(mirna_feature_stack)
        mirna_emb = self.dropout(mirna_emb)

        combined_features = torch.cat([drug_emb, mirna_emb], dim=1)
        output = self.classifier_fc(combined_features)

        return output


