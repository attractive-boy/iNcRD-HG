# -*- coding: utf-8 -*-
"""Baseline network definitions, adapted from each method's official implementation
to the miRNA-drug binary-classification setting of the DLST-MDA benchmark.

All models return a single logit (BCEWithLogitsLoss).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv, global_max_pool as gmp, global_add_pool

from common import smile_to_graph, seq2int, smiles2int, SEQ_LEN

ATOM_DIM = 78


# =====================================================================================
# 1) GraphDTA (Nguyen et al., Bioinformatics 2021) -- GCN drug encoder + 1D-CNN miRNA
#    Adapted: protein branch re-sized for 24-nt miRNA (official hard-codes len 1000).
# =====================================================================================
class GraphDTA_GCN(nn.Module):
    def __init__(self, num_features_xd=ATOM_DIM, n_filters=32, embed_dim=128,
                 output_dim=128, dropout=0.2, seq_len=SEQ_LEN):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # drug graph branch
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.fc_g1 = nn.Linear(num_features_xd * 4, 1024)
        self.fc_g2 = nn.Linear(1024, output_dim)
        # miRNA sequence branch (single conv, kernel 8, as in official GraphDTA)
        self.embedding_xt = nn.Embedding(len(range(5)) + 1, embed_dim)
        self.conv_xt = nn.Conv1d(embed_dim, n_filters, kernel_size=8)
        conv_len = seq_len - 8 + 1
        self.fc_xt = nn.Linear(n_filters * conv_len, output_dim)
        # combined
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)

    def encode_drug(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.relu(self.conv3(x, edge_index))
        x = gmp(x, batch)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)
        return x

    def encode_rna(self, target):
        e = self.embedding_xt(target).permute(0, 2, 1)  # B, emb, L
        e = self.relu(self.conv_xt(e))
        e = e.flatten(1)
        return self.relu(self.fc_xt(e))

    def forward(self, data):
        x = self.encode_drug(data)
        xt = self.encode_rna(data.target)
        xc = torch.cat((x, xt), 1)
        xc = self.relu(self.fc1(xc))
        xc = self.dropout(xc)
        xc = self.relu(self.fc2(xc))
        xc = self.dropout(xc)
        return self.out(xc)


# =====================================================================================
# 2) SubMDTA (Pan et al., BMC Bioinformatics 2023) -- GIN drug encoder + multi-scale
#    BiLSTM miRNA encoder.  NOTE: the substructure self-supervised pre-training stage
#    is omitted (documented); the DTA prediction architecture is kept.
# =====================================================================================
class _BiLSTMBlock(nn.Module):
    """n-gram embedding -> BiLSTM -> adaptive max pool -> 2*hidden feature."""
    def __init__(self, vocab, emb=128, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)

    def forward(self, x):
        e = self.embed(x)
        h, _ = self.lstm(e)
        return h.max(dim=1).values  # B, 2*hidden


class SubMDTA(nn.Module):
    def __init__(self, num_features_xd=ATOM_DIM, dim=128, output_dim=128, dropout=0.2,
                 seq_len=SEQ_LEN):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # drug: 3-layer GIN (SubMDTA uses GIN encoder)
        self.gin1 = GINConv(Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim)))
        self.gin2 = GINConv(Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim)))
        self.gin3 = GINConv(Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim)))
        self.fc_x = nn.Linear(dim, output_dim)
        # mirna: multi-scale (2,3,4)-mer BiLSTM
        self.scales = [2, 3, 4]
        self.ngram_vocab = {n: (4 ** n) + 1 for n in self.scales}
        self.blocks = nn.ModuleList([_BiLSTMBlock(self.ngram_vocab[n]) for n in self.scales])
        self.fc_xt = nn.Linear(2 * 64 * len(self.scales), output_dim)
        # combined
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, 1)

    def _ngrams(self, target):
        # target: B, L ints in 0..4 (0 = padding). base-4 codes shifted by +1.
        b = (target - 1).clamp(min=0)  # 0..3, padding -> 0
        L = target.size(1)
        outs = []
        for n in self.scales:
            idx = b[:, :L - n + 1].clone()
            for k in range(1, n):
                idx = idx * 4 + b[:, k:L - n + 1 + k]
            outs.append(idx + 1)
        return outs

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relu(self.gin1(x, edge_index))
        x = self.relu(self.gin2(x, edge_index))
        x = self.relu(self.gin3(x, edge_index))
        x = global_add_pool(x, batch)
        x = self.relu(self.fc_x(x))
        num_graphs = x.size(0)
        outs = [blk(g) for blk, g in zip(self.blocks, self._ngrams(data.target))]
        xt = self.relu(self.fc_xt(torch.cat(outs, dim=-1)))
        xc = torch.cat((x, xt), 1)
        xc = self.relu(self.fc1(xc))
        xc = self.dropout(xc)
        xc = self.relu(self.fc2(xc))
        xc = self.dropout(xc)
        return self.out(xc)


# =====================================================================================
# 3) ML-DTI (Yang et al., J. Phys. Chem. Lett. 2021) -- mutual-attention dual CNN.
#    Copied from official network/ML_DTI.py; protein slot <- miRNA, ligand <- SMILES.
# =====================================================================================
class _Conv1dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding), nn.ReLU())

    def forward(self, x):
        return self.inc(x)


class _LinearReLU(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.inc = nn.Sequential(nn.Linear(in_features, out_features, bias=bias), nn.ReLU())

    def forward(self, x):
        return self.inc(x)


class _BilinearPooling(nn.Module):
    def __init__(self, in_channels, out_channels, c_m, c_n):
        super().__init__()
        self.convA = nn.Conv1d(in_channels, c_m, kernel_size=1)
        self.convB = nn.Conv1d(in_channels, c_n, kernel_size=1)
        self.linear = nn.Linear(c_m, out_channels)

    def forward(self, x):
        A = self.convA(x)
        B = self.convB(x)
        att = F.softmax(B, dim=-1)
        gd = torch.bmm(A, att.permute(0, 2, 1))
        gd = torch.mean(gd, dim=-1)
        return self.linear(gd).unsqueeze(1)


class _MutualAttention(nn.Module):
    def __init__(self, in_channels, att_size, c_m, c_n):
        super().__init__()
        self.bipool = _BilinearPooling(in_channels, in_channels, c_m, c_n)
        self.linearS = nn.Linear(in_channels, att_size)
        self.linearT = nn.Linear(in_channels, att_size)

    def forward(self, source, target):
        gd = self.bipool(source)
        target_org = target
        target = self.linearT(target.permute(0, 2, 1)).permute(0, 2, 1)
        gd = self.linearS(gd)
        att = torch.bmm(gd, target)
        att = torch.sigmoid(att)
        out = torch.add(target_org, torch.mul(target_org, att))
        return F.relu(out)


class MLDTI(nn.Module):
    def __init__(self, vocab_prot_size=5, vocab_drug_size=67, embedding_size=128, filter_num=32):
        super().__init__()
        pf = [4, 8, 12]
        df = [4, 6, 8]
        self.prot_embed = nn.Embedding(vocab_prot_size, embedding_size, padding_idx=0)
        self.prot_conv1 = _Conv1dReLU(embedding_size, filter_num, pf[0])
        self.prot_conv2 = _Conv1dReLU(filter_num, filter_num * 2, pf[1])
        self.prot_conv3 = _Conv1dReLU(filter_num * 2, filter_num * 3, pf[2])
        self.prot_pool = nn.AdaptiveMaxPool1d(1)
        self.drug_embed = nn.Embedding(vocab_drug_size, embedding_size, padding_idx=0)
        self.drug_conv1 = _Conv1dReLU(embedding_size, filter_num, df[0])
        self.drug_conv2 = _Conv1dReLU(filter_num, filter_num * 2, df[1])
        self.drug_conv3 = _Conv1dReLU(filter_num * 2, filter_num * 3, df[2])
        self.drug_pool = nn.AdaptiveMaxPool1d(1)
        self.prot_mut_att1 = _MutualAttention(filter_num, filter_num, filter_num, 8)
        self.drug_mut_att1 = _MutualAttention(filter_num, filter_num, filter_num, 8)
        self.prot_mut_att2 = _MutualAttention(filter_num * 2, filter_num, filter_num, 8)
        self.drug_mut_att2 = _MutualAttention(filter_num * 2, filter_num, filter_num, 8)
        self.prot_mut_att3 = _MutualAttention(filter_num * 3, filter_num, filter_num, 8)
        self.drug_mut_att3 = _MutualAttention(filter_num * 3, filter_num, filter_num, 8)
        self.linear1 = _LinearReLU(filter_num * 3 * 2, 1024)
        self.drop1 = nn.Dropout(0.1)
        self.linear2 = _LinearReLU(1024, 1024)
        self.drop2 = nn.Dropout(0.1)
        self.linear3 = _LinearReLU(1024, 512)
        self.drop3 = nn.Dropout(0.1)
        self.out_layer = nn.Linear(512, 1)

    def forward(self, prot_x, drug_x):
        prot_x = self.prot_embed(prot_x).permute(0, 2, 1)
        drug_x = self.drug_embed(drug_x).permute(0, 2, 1)
        prot_x = self.prot_conv1(prot_x)
        drug_x = self.drug_conv1(drug_x)
        prot_g = self.prot_mut_att1(drug_x, prot_x)
        drug_g = self.drug_mut_att1(prot_x, drug_x)
        prot_x = self.prot_conv2(prot_g)
        drug_x = self.drug_conv2(drug_g)
        prot_g = self.prot_mut_att2(drug_x, prot_x)
        drug_g = self.drug_mut_att2(prot_x, drug_x)
        prot_x = self.prot_conv3(prot_g)
        drug_x = self.drug_conv3(drug_g)
        prot_g = self.prot_mut_att3(drug_x, prot_x)
        drug_g = self.drug_mut_att3(prot_x, drug_x)
        prot_x = self.prot_pool(prot_g).squeeze(-1)
        drug_x = self.drug_pool(drug_g).squeeze(-1)
        x = torch.cat([prot_x, drug_x], dim=-1)
        x = self.drop1(self.linear1(x))
        x = self.drop2(self.linear2(x))
        x = self.drop3(self.linear3(x))
        return self.out_layer(x)


# =====================================================================================
# 4) GCNNMMA / SMiR-GCNN (Niu et al., Frontiers in Genetics 2023)
#    Official model (verbatim structure). Inputs come from the official preprocessor.
# =====================================================================================
def build_gcnnma(dim=10, layer_gnn=3, window=11, layer_cnn=3, layer_output=3):
    import importlib.util
    spec = importlib.util.spec_from_file_location("missing", __file__)  # placeholder
    raise NotImplementedError


# =====================================================================================
# PyG dataset helper for graph models
# =====================================================================================
def build_graph_dataset(pairs, labels, device='cpu'):
    """pairs: list of (smiles, mirna_seq). Returns list[Data]."""
    data_list = []
    for (smiles, seq), y in zip(pairs, labels):
        _, features, edge_index = smile_to_graph(smiles)
        d = Data(x=torch.tensor(np.asarray(features), dtype=torch.float),
                 edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                 y=torch.tensor([float(y)], dtype=torch.float))
        d.target = torch.tensor(seq2int(seq), dtype=torch.long).unsqueeze(0)
        if device != 'cpu':
            d = d.to(device)
        data_list.append(d)
    return data_list
