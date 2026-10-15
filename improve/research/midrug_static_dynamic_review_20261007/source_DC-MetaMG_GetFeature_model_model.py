import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.utils import InputType
from torch_geometric.nn import GCNConv
from model.DrugMG import *
from util.util import *

class Channel(torch.nn.Module):
    input_type = InputType.PAIRWISE

    def __init__(self, hidden_dim, n_layers):
        super(Channel, self).__init__()

        self.device = torch.device("cuda:0")

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.layers = nn.ModuleList()

        self.res_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ELU()

        self.gcn_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.gcn_layers.append(GCNConv(in_channels=self.hidden_dim,
                                           out_channels=self.hidden_dim))

            self.mlp_layers.append(nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ))

    def forward(self, x, edge_index):
        x = x.to(device=self.device)
        edge_index = edge_index.to(device=self.device)
        res = self.res_proj(x)
        for i, layer in enumerate(self.gcn_layers):
            x = self.dropout(x)

            x = layer(x, edge_index)
            x = self.mlp_layers[i](x)

            x = self.activation(x + res)
            #x = x + layer(x, edge_index)
        # x = self.mlp(x)
        return x


class MetaMG(torch.nn.Module):
    input_type = InputType.PAIRWISE

    def __init__(self, args, smiles, num_hidden, num_layer, dataset):
        super(MetaMG, self).__init__()

        # load parameters info
        self.device = torch.device("cuda:0")
        self.args = args

        self.embedding_size = self.args.embedding_size
        self.hidden_dim = self.args.embedding_size
        self.n_layers = self.args.n_layers

        self.NumDrug = self.args.n_Drug
        self.NumNcRNA = self.args.n_NcRNA

        self.temperature = args.temperature

        self.alpha = args.alpha
        self.dataset = dataset

        self.DrugMG = DrugMG(smiles, num_hidden, num_layer).to(args.device)
        self.InitFeature()

        self.Channel1 = Channel(self.hidden_dim, self.n_layers).to(args.device)
        self.Channel2 = Channel(self.hidden_dim, self.n_layers).to(args.device)

    def InitFeature(self):

        self.Drug_GT = torch.nn.Embedding(self.args.n_Drug,
                                          self.embedding_size)

        self.NcRNA_GS = torch.nn.Embedding(self.args.n_NcRNA,
                                           self.embedding_size)

        self.Drug_MG = torch.nn.Embedding(self.args.n_Drug,
                                          self.embedding_size)

        self.NcRNA_SQ = torch.nn.Embedding(self.args.n_NcRNA,
                                           self.embedding_size)


        Drug_MG = self.DrugMG.getMGemb()

        Drug_GT = np.loadtxt(os.path.join(self.args.data_path, 'GIP_Topo_drug.txt'), dtype=float, delimiter=" ")
        NcRNA_GS = np.loadtxt(os.path.join(self.args.data_path, f'Gaussian_sim_{self.args.dataset}.txt'), dtype=float, delimiter=" ")
        NcRNA_SQ = np.loadtxt(os.path.join(self.args.data_path, f'Feature_{self.args.dataset}_SQ.txt'), dtype=float, delimiter=None)


        Drug_GT = torch.from_numpy(Drug_GT).float().to(self.device)
        Drug_MG = torch.from_numpy(Drug_MG).float().to(self.device)
        NcRNA_GS = torch.from_numpy(NcRNA_GS).float().to(self.device)
        NcRNA_SQ = torch.from_numpy(NcRNA_SQ).float().to(self.device)

        with torch.no_grad():
            self.Drug_GT.weight.copy_(Drug_GT)
            self.NcRNA_GS.weight.copy_(NcRNA_GS)
            self.Drug_MG.weight.copy_(Drug_MG)
            self.NcRNA_SQ.weight.copy_(NcRNA_SQ)

    def get_embeddings(self):
        DrugGT_emb = self.Drug_GT.weight
        NcRNAGS_emb = self.NcRNA_GS.weight
        embeddings1 = torch.cat([DrugGT_emb, NcRNAGS_emb], dim=0)

        DrugMG_emb = self.Drug_MG.weight
        NcRNASQ_emb = self.NcRNA_SQ.weight
        embeddings2 = torch.cat([DrugMG_emb, NcRNASQ_emb], dim=0)
        return embeddings1, embeddings2

    def forward(self, ):

        all_embeddings1, all_embeddings2 = self.get_embeddings()
        embeddings_list1 = [all_embeddings1]
        embeddings_list2 = [all_embeddings2]


        edge_index1 = get_edge_index(self.args, self.dataset, 'MetaPath')
        edge_index2 = get_edge_index(self.args, self.dataset, )

        channel1 = self.Channel1(all_embeddings1, edge_index1)
        channel2 = self.Channel2(all_embeddings2, edge_index2)

        embeddings_list1.append(channel1)
        embeddings_list2.append(channel2)
        embeddings1_list = torch.stack(embeddings_list1, dim=0)
        embeddings2_list = torch.stack(embeddings_list2, dim=0)

        drug_all_embeddings, mirna_all_embeddings = torch.split(channel1,
                                                                [self.NumDrug, self.NumNcRNA])
        drug2_all_embeddings, mirna2_all_embeddings = torch.split(channel2,
                                                                  [self.NumDrug, self.NumNcRNA])
        left_emb = torch.concat([drug_all_embeddings.unsqueeze(dim=0), drug2_all_embeddings.unsqueeze(dim=0)], dim=0)
        right_emb = torch.concat([mirna_all_embeddings.unsqueeze(dim=0), mirna2_all_embeddings.unsqueeze(dim=0)], dim=0)

        embeddings_all = torch.stack((embeddings1_list, embeddings2_list), dim=0)
        return left_emb, right_emb, embeddings_all

    def predict(self, ):

        drug_all_embeddings, ncRNA_all_embeddings, _ = self.forward()
        drug_embeddings = drug_all_embeddings[:, 0:self.NumDrug].cpu().detach().numpy().tolist()
        ncRNA_embeddings = ncRNA_all_embeddings[:, 0:self.NumNcRNA].cpu().detach().numpy().tolist()

        CoarseGrained_Drug = []
        FineGrained_Drug = []
        CoarseGrained_RNA = []
        FineGrained_RNA = []

        for i in range(len(drug_embeddings[0])):
            CoarseGrained_Drug.append(drug_embeddings[0][i])
            FineGrained_Drug.append(drug_embeddings[1][i])

        for i in range(len(ncRNA_embeddings[0])):
            CoarseGrained_RNA.append(ncRNA_embeddings[0][i])
            FineGrained_RNA.append(ncRNA_embeddings[1][i])

        return CoarseGrained_Drug, CoarseGrained_RNA, FineGrained_Drug, FineGrained_RNA

    def positive_similarity_loss(self, lastlayer_emb, firstlayer_emb, pos_drug_idx, pos_mirna_idx):

        drug_emb = F.normalize(lastlayer_emb[pos_drug_idx])  # [num_pos, dim]
        mirna_emb = F.normalize(firstlayer_emb[pos_mirna_idx])  # [num_pos, dim]


        sim = torch.mul(drug_emb, mirna_emb).sum(dim=-1)  # [num_pos]


        loss = (1 - sim).mean()

        return loss

    def calculate_loss(self, train_data):

        posINDEX_drug = []
        posINDEX_mirna = []
        negINDEX_drug = []
        negINDEX_mirna = []


        for d, m in zip(train_data.neg_edge_label_index[0], train_data.neg_edge_label_index[1]):
            negINDEX_drug.append(int(d))
            negINDEX_mirna.append(int(m))

        for d, m in zip(train_data.pos_edge_label_index[0], train_data.pos_edge_label_index[1]):
            posINDEX_drug.append(int(d))
            posINDEX_mirna.append(int(m))

        drug_all_embeddings, mirna_all_embeddings, embeddings_list = self.forward()


        info_nce_LossD1 = self.infoNCE_Loss(
            embeddings_list[0][-1][:],
            embeddings_list[0][0][:],
            posINDEX_drug,
            negINDEX_drug,
            self.temperature
        )

        info_nce_LossM1 = self.infoNCE_Loss(
            embeddings_list[0][-1][:],
            embeddings_list[0][0][:],
            posINDEX_mirna,
            negINDEX_mirna,
            self.temperature
        )

        info_nce_LossD2 = self.infoNCE_Loss(
            embeddings_list[1][-1][:],
            embeddings_list[1][0][:],
            posINDEX_drug,
            negINDEX_drug,
            self.temperature
        )

        info_nce_LossM2 = self.infoNCE_Loss(
            embeddings_list[1][-1][:],
            embeddings_list[1][0][:],
            posINDEX_mirna,
            negINDEX_mirna,
            self.temperature
        )


        pos_sim_loss = self.positive_similarity_loss(
            embeddings_list[0][-1][:],
            embeddings_list[0][0][:],
            posINDEX_drug,
            posINDEX_mirna
        )

        total_loss = (info_nce_LossD1 + info_nce_LossD2 +
                      info_nce_LossM1 + info_nce_LossM2 +
                      0.1 * pos_sim_loss)

        return total_loss

    def infoNCE_Loss(self, lastlayer_emb, firstlayer_emb, pos_index, neg_index, temperature=0.1):

        posemb_last = F.normalize(lastlayer_emb[pos_index])  # [num_pos, dim]
        posemb_first = F.normalize(firstlayer_emb[pos_index])  # [num_pos, dim]
        negemb_first = F.normalize(firstlayer_emb[neg_index])  # [num_neg, dim]


        pos_sim = torch.mul(posemb_last, posemb_first).sum(dim=-1)  # [num_pos]

        neg_sim = torch.matmul(posemb_last, negemb_first.T)  # [num_pos, num_neg]

        pos_exp = torch.exp(pos_sim / temperature)  # [num_pos]

        neg_exp = torch.exp(neg_sim / temperature)  # [num_pos, num_neg]
        denom = pos_exp.unsqueeze(1) + neg_exp.sum(dim=1, keepdim=True)  # [num_pos, 1]

        loss = -torch.log(pos_exp.unsqueeze(1) / denom).mean()

        return loss



