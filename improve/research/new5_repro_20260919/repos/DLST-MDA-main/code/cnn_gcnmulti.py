import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp

# GCN based model
class GCNNetmuti(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=64, num_features_xd=78, num_features_smile=66, num_features_xt=25, output_dim=128, dropout=0.2):
        super(GCNNetmuti, self).__init__()

        # SMILES character CNN processing
        self.smile_embed = nn.Embedding(num_features_smile + 1, embed_dim)
        self.conv_xd_11 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=3, padding=1)
        self.conv_xd_12 = nn.Conv1d(n_filters, out_channels=n_filters*2, kernel_size=3, padding=1)
        self.conv_xd_21 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=2, padding=1)
        self.conv_xd_22 = nn.Conv1d(n_filters, out_channels=n_filters*2, kernel_size=2, padding=1)
        self.conv_xd_31 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=1, padding=1)
        self.conv_xd_32 = nn.Conv1d(n_filters, out_channels=n_filters*2, kernel_size=1, padding=1)
        self.fc_smiles = torch.nn.Linear(n_filters * 2, output_dim)

        # SMILES graph branch
        self.n_output = n_output
        self.gcnv1 = GCNConv(num_features_xd, num_features_xd*2)
        self.gcnv2 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Convolution layers for reducing dimensions after concatenation
        self.conv_reduce_smiles = nn.Conv1d(in_channels=output_dim*3, out_channels=output_dim, kernel_size=1)
        self.conv_reduce_xt = nn.Conv1d(in_channels=192, out_channels=output_dim, kernel_size=1)

        self.conv_transform = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=1)

        # miRNA sequence branch (1D CNN)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_11 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=4, padding=2)
        self.conv_xt_12 = nn.Conv1d(n_filters, out_channels=n_filters*2, kernel_size=4, padding=2)

        self.conv_xt_21 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=3, padding=1)
        self.conv_xt_22 = nn.Conv1d(n_filters, out_channels=n_filters*2, kernel_size=3, padding=1)

        self.conv_xt_31 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=2, padding=1)
        self.conv_xt_32 = nn.Conv1d(n_filters, out_channels=n_filters*2, kernel_size=2, padding=1)

        self.fc1_xt = nn.Linear(n_filters * 2, output_dim)

        # Combined layers
        self.fc1 = nn.Linear(output_dim * 2, 256)
        self.out = nn.Linear(256, self.n_output)
        self.ac = nn.Sigmoid()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        drugsmile = data.seqdrug
        target = data.target

        x = self.gcnv1(x, edge_index)
        x = self.relu(x)
        x = self.gcnv2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)
        x = self.fc_g1(x)
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)


        embedded_smile = self.smile_embed(drugsmile.long())
        embedded_smile = embedded_smile.permute(0, 2, 1)

        conv_xd1 = self.conv_xd_11(embedded_smile)
        conv_xd1 = self.relu(conv_xd1)
        conv_xd1=self.dropout(conv_xd1)
        conv_xd1 = F.max_pool1d(conv_xd1, kernel_size=2)



        conv_xd1 = self.conv_xd_12(conv_xd1)
        conv_xd1 = self.relu(conv_xd1)

        conv_xd1 = F.max_pool1d(conv_xd1, conv_xd1.size(2)).squeeze(2)

        conv_xd2 = self.conv_xd_21(embedded_smile)
        conv_xd2 = self.relu(conv_xd2)
        conv_xd2 = self.dropout(conv_xd2)
        conv_xd2 = F.max_pool1d(conv_xd2, kernel_size=2)


        conv_xd2 = self.conv_xd_22(conv_xd2)
        conv_xd2 = self.relu(conv_xd2)
        conv_xd2 = F.max_pool1d(conv_xd2, conv_xd2.size(2)).squeeze(2)

        conv_xd3 = self.conv_xd_31(embedded_smile)
        conv_xd3 = self.relu(conv_xd3)
        conv_xd3=self.dropout(conv_xd3)
        conv_xd3 =  F.max_pool1d(conv_xd3, kernel_size=2)



        conv_xd3 = self.conv_xd_32(conv_xd3)
        conv_xd3 = self.relu(conv_xd3)
        conv_xd3 =F.max_pool1d(conv_xd3, conv_xd3.size(2)).squeeze(2)


        conv_xd1 = self.fc_smiles(conv_xd1)
        conv_xd2 = self.fc_smiles(conv_xd2)
        conv_xd3 = self.fc_smiles(conv_xd3)


        conv_xd = torch.cat((conv_xd1, conv_xd2, conv_xd3), dim=1)
        conv_xd = conv_xd.unsqueeze(1).permute(0, 2, 1)

        conv_xd = self.conv_reduce_smiles(conv_xd)
        conv_xd=conv_xd.squeeze(2)

        x=conv_xd+x
        embedded_xt = self.embedding_xt(target)
        embedded_xt = embedded_xt.permute(0, 2, 1)

        conv_xt1 = self.conv_xt_11(embedded_xt)
        conv_xt1 = self.relu(conv_xt1)
        conv_xt1=self.dropout(conv_xt1)
        conv_xt1 = self.conv_xt_12(conv_xt1)
        conv_xt1 = self.relu(conv_xt1)
        conv_xt1 = F.max_pool1d(conv_xt1, conv_xt1.size(2)).squeeze(2)

        conv_xt2 = self.conv_xt_21(embedded_xt)
        conv_xt2 = self.relu(conv_xt2)

        conv_xt2 = self.dropout(conv_xt2)
        conv_xt2 = self.conv_xt_22(conv_xt2)
        conv_xt2 = self.relu(conv_xt2)
        conv_xt2 =F.max_pool1d(conv_xt2, conv_xt2.size(2)).squeeze(2)

        conv_xt3 = self.conv_xt_31(embedded_xt)
        conv_xt3 = self.relu(conv_xt3)
        conv_xt3 = F.max_pool1d(conv_xt3, kernel_size=2)
        conv_xt3 = self.dropout(conv_xt3)
        conv_xt3 = self.conv_xt_32(conv_xt3)
        conv_xt3 = self.relu(conv_xt3)
        conv_xt3 = F.max_pool1d(conv_xt3, conv_xt3.size(2)).squeeze(2)


        conv_xt = torch.cat((conv_xt1, conv_xt2, conv_xt3), dim=1)

        conv_xt = conv_xt.unsqueeze(2)
        conv_xt = self.conv_reduce_xt(conv_xt)
        conv_xt=conv_xt.squeeze(2)

        xc=torch.cat((x, conv_xt), dim=1)

        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.ac(out)
        return out
