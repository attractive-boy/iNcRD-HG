import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,  global_mean_pool, global_max_pool
from torch_geometric.data import Batch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import fm
from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch_geometric')

class CNNFeatureExtractor(nn.Module):

    def __init__(self, output_dim, vocab_size=5, embed_dim=128, num_filters=64, filter_sizes=[2, 3, 4]):
        super(CNNFeatureExtractor, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch_size, embed_dim, seq_len)

        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch_size, num_filters, conv_seq_len)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch_size, num_filters)
            conv_outputs.append(pooled)

        x = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class RNAFMFeatureExtractor(nn.Module):

    def __init__(self, output_dim, model_layer=12):
        super(RNAFMFeatureExtractor, self).__init__()

        self.model, self.alphabet = fm.pretrained.rna_fm_t12()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.model_layer = model_layer
        self.rna_fm_dim = 640

        self.projection = nn.Linear(self.rna_fm_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, rna_sequence):

        if isinstance(rna_sequence, list) and isinstance(rna_sequence[0], str):
            rna_sequence = [(f"seq_{i}", seq) for i, seq in enumerate(rna_sequence)]

        device = next(self.projection.parameters()).device
        batch_labels, batch_strs, batch_tokens = self.batch_converter(rna_sequence)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.model_layer])
            token_embeddings = results["representations"][self.model_layer]

            # sequence_embeddings = token_embeddings[:, 1:-1, :]  # (batch, seq_len, rna_fm_dim)
            # pooled_embeddings = sequence_embeddings.mean(dim=1)  # (batch, rna_fm_dim)
            cls_token_embeddings = token_embeddings[:, 0, :]  # (batch, rna_fm_dim)

        features = self.projection(cls_token_embeddings)
        features = self.dropout(features)

        return features


class GCNFeatureExtractor(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(GCNFeatureExtractor, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.dropout = nn.Dropout(0.2)

        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)])
        self.batch_norm.append(nn.BatchNorm1d(output_dim))


    def forward(self, x, edge_index, batch):

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            x = self.batch_norm[i](x)

            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        x = global_mean_pool(x, batch) + global_max_pool(x, batch)
        return x

class ChemBERTaFeatureExtractor(nn.Module):

    def __init__(self, output_dim):
        super(ChemBERTaFeatureExtractor, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        self.model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

        for param in self.model.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(self.model.config.hidden_size, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, smiles_list):

        encoded = self.tokenizer(smiles_list,
                                 padding=True,
                                 truncation=True,
                                 max_length=512,
                                 return_tensors='pt')

        encoded = {k: v.to(next(self.model.parameters()).device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        # print("cls_embeddings", cls_embeddings.shape)

        # sequence_embeddings = outputs.last_hidden_state[:, 1:-1, :]  # (batch, seq_len, rna_fm_dim)
        # pooled_embeddings = sequence_embeddings.mean(dim=1)  # (batch, rna_fm_dim)

        features = self.projection(cls_embeddings)
        features = self.dropout(features)

        return features

class CrossAttention(nn.Module):

    def __init__(self, d_model, num_heads=2):

        super(CrossAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) # (bs, num_heads, seq_len_q, head_dim)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # print("Q,K,V", Q.shape, K.shape, V.shape)  # torch.Size([32, 2, 1, 128])

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.out_proj(context)
        return output

class FeatureFusion(nn.Module):

    def __init__(self, feature_dim):
        super(FeatureFusion, self).__init__()
        self.cross_attention = CrossAttention(feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, f1, f2):

        f1_expanded = f1.unsqueeze(1)  # (batch_size, 1, feature_dim)
        f2_expanded = f2.unsqueeze(1)  # (batch_size, 1, feature_dim)

        attended_f1 = self.cross_attention(f1_expanded, f2_expanded, f2_expanded)
        attended_f2 = self.cross_attention(f2_expanded, f1_expanded, f1_expanded)

        f1_out = self.layer_norm(f1_expanded + attended_f1)
        f2_out = self.layer_norm(f2_expanded + attended_f2)

        fused = (f1_out + f2_out).squeeze(1)  # (batch_size, feature_dim)

        return fused

class InteractionPredictor(nn.Module):

    def __init__(self, hidden_dim):
        super(InteractionPredictor, self).__init__()

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, mirna_features, drug_features):

        combined = torch.cat([mirna_features, drug_features], dim=1)
        score = self.predictor(combined)

        return score.squeeze(-1)


class DrugMiRNAInteractionModel(nn.Module):

    def __init__(self, args):
        super(DrugMiRNAInteractionModel, self).__init__()

        #miRNA feature extractor
        self.mirna_cnn = CNNFeatureExtractor(output_dim=args.mirna_feature_dim)
        self.mirna_rnafm = RNAFMFeatureExtractor(output_dim=args.mirna_feature_dim)

        # drug feature extractor
        self.drug_gcn = GCNFeatureExtractor(
            input_dim=args.drug_input_dim,
            hidden_dim=args.drug_hidden_dim,
            output_dim=args.drug_feature_dim,
        )
        self.drug_chembert = ChemBERTaFeatureExtractor(output_dim=args.drug_feature_dim)

        # feature fusion
        self.mirna_fusion = FeatureFusion(feature_dim=args.mirna_feature_dim)
        self.drug_fusion = FeatureFusion(feature_dim=args.drug_feature_dim)

        # prediction
        self.interaction_predictor = InteractionPredictor(hidden_dim=args.mirna_feature_dim + args.drug_feature_dim)

    def forward(self, drug_graphs, drug_smiles, mirna_encoded, mirna_seqs):
        device = next(self.parameters()).device

        if isinstance(drug_graphs, list):
            drug_batch = Batch.from_data_list(drug_graphs)
        else:
            drug_batch = drug_graphs

        drug_batch = drug_batch.to(device)
        mirna_encoded = mirna_encoded.to(device)

        m1 = self.mirna_cnn(mirna_encoded)
        m2 = self.mirna_rnafm(mirna_seqs)

        d1 = self.drug_gcn(drug_batch.x, drug_batch.edge_index, drug_batch.batch)
        d2 = self.drug_chembert(drug_smiles)

        mirna_features = self.mirna_fusion(m1, m2)
        drug_features = self.drug_fusion(d1, d2)

        interaction_score = self.interaction_predictor(mirna_features, drug_features)

        return interaction_score