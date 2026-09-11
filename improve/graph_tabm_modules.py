"""Graph-derived pair features and a compact TabM-style network.

TabM: Gorishniy et al., ICLR 2025, arXiv:2410.24210. The network uses
BatchEnsemble layers, random signs only on the first input scaling, unit
initialization for other scalings, and independent output heads. It is an
internal ensemble, not a single MLP and not an ensemble of separately trained
GNNs. This small implementation supports the workspace's PyTorch 1.13.
"""
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
import torch
from torch import nn


class PairFeatureEncoder:
    """All target-label-dependent features use the supplied support positives.

    Fixed side information comes from the same graph/features as the GNN.
    No external similarity matrix, validation label or new data source is used.
    """

    def __init__(self, graph, task):
        self.task = task
        self.rna = 'lncRNA' if task == 'LncDrug' else 'miRNA'
        self.other = 'miRNA' if task == 'LncDrug' else 'lncRNA'
        self.other_task = 'MiDrug' if task == 'LncDrug' else 'LncDrug'
        self.nr, self.nd = graph[self.rna].num_nodes, graph['drug'].num_nodes

        def dense(key):
            src, _, dst = key
            a = np.zeros((graph[src].num_nodes, graph[dst].num_nodes), dtype=np.float32)
            edge = graph[key].edge_index.cpu().numpy()
            a[edge[0], edge[1]] = 1.
            return a

        self.sr = dense((self.rna, 'LncLnc' if task == 'LncDrug' else 'MiMi', self.rna))
        self.sd = dense(('drug', 'DrugDrug', 'drug'))
        np.fill_diagonal(self.sr, 0.)
        np.fill_diagonal(self.sd, 0.)
        link = dense(('lncRNA', 'LncMi', 'miRNA'))
        other = dense((self.other, self.other_task, 'drug'))
        self.bridge = (link if task == 'LncDrug' else link.T) @ other
        self.xr = graph[self.rna].x.cpu().numpy()
        self.xd = graph['drug'].x.cpu().numpy()
        self.ur = self.unit(self.xr)
        self.ud = self.unit(self.xd)
        self.side = {}
        target = {(self.rna, task, 'drug'), ('drug', 'rev_'+task, self.rna)}
        for i, (key, edge) in enumerate(graph.edge_index_dict.items()):
            if key in target:
                continue
            dst = key[2]
            if dst in (self.rna, 'drug'):
                values = np.bincount(edge[1].cpu().numpy(), minlength=graph[dst].num_nodes)
                self.side['degree_'+str(i)] = (dst, values)

    @staticmethod
    def unit(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

    def transform(self, pairs, support_positive):
        pairs = np.asarray(pairs, dtype=np.int64)
        support_positive = np.asarray(support_positive, dtype=np.int64)
        a = np.zeros((self.nr, self.nd), dtype=np.float32)
        a[support_positive[0], support_positive[1]] = 1.
        r, d = pairs
        if np.any(a[r, d]):
            raise ValueError('A query positive is visible to its own feature encoder')
        dr, dd = a.sum(1), a.sum(0)
        sr_a, a_sd = self.sr @ a, a @ self.sd
        gram = a.T @ a
        a3 = a @ gram
        a5 = a3 @ gram
        u, sig, vt = np.linalg.svd(a, full_matrices=False)
        rank = min(16, len(sig))
        factor = (u[:, :rank] * sig[:rank]) @ vt[:rank]
        r_neighbor = (a @ dd) / np.maximum(dr, 1)
        d_neighbor = (a.T @ dr) / np.maximum(dd, 1)
        attribute_d = self.unit(a @ self.ud)
        attribute_r = self.unit(a.T @ self.ur)
        expr_support = (self.ur @ self.ur.T)
        drug_support = (self.ud @ self.ud.T)
        np.fill_diagonal(expr_support, 0.)
        np.fill_diagonal(drug_support, 0.)
        cols = {
            'degree_r': dr[r], 'degree_d': dd[d], 'degree_product': dr[r]*dd[d],
            'collab_r': sr_a[r, d], 'collab_d': a_sd[r, d],
            'collab_r_mean': sr_a[r, d] / np.maximum(self.sr.sum(1)[r], 1),
            'collab_d_mean': a_sd[r, d] / np.maximum(self.sd.sum(0)[d], 1),
            'collab_both': (sr_a @ self.sd)[r, d],
            'path3': a3[r, d], 'path3_norm': a3[r, d]/np.maximum(dr[r]*dd[d], 1),
            'path5': a5[r, d], 'path5_norm': a5[r, d]/np.maximum(dr[r]*dd[d], 1),
            'svd16': factor[r, d], 'neighbor_degree_r': r_neighbor[r],
            'neighbor_degree_d': d_neighbor[d], 'bridge': self.bridge[r, d],
            'bridge_norm': self.bridge[r, d] / np.maximum(dd[d], 1),
            'rna_similarity_degree': (self.sr @ dr)[r],
            'drug_similarity_degree': (self.sd @ dd)[d],
            'collab_r2': (self.sr @ sr_a)[r, d],
            'collab_d2': (a_sd @ self.sd)[r, d],
            'rna_attribute_support_sum': (expr_support @ a)[r, d],
            'drug_attribute_support_sum': (a @ drug_support)[r, d],
            'drug_attribute_profile_cos': (attribute_d[r]*self.ud[d]).sum(1),
            'rna_attribute_profile_cos': (attribute_r[d]*self.ur[r]).sum(1),
            'rna_feature_mean': self.xr.mean(1)[r], 'rna_feature_std': self.xr.std(1)[r],
            'rna_feature_norm': np.linalg.norm(self.xr, axis=1)[r],
            'drug_feature_mean': self.xd.mean(1)[d], 'drug_feature_std': self.xd.std(1)[d],
            'drug_feature_norm': np.linalg.norm(self.xd, axis=1)[d],
        }
        for name, (nt, values) in self.side.items():
            cols[name] = values[r if nt == self.rna else d]
        frame = pd.DataFrame(cols, dtype=np.float32)
        if not np.isfinite(frame.values).all():
            raise ValueError('Non-finite pair features')
        return frame


def crossfit_features(graph, task, pairs, labels, evaluation_pairs, folds=5, seed=20260910):
    """Fit features out-of-fold; evaluation features use all fitting positives."""
    encoder = PairFeatureEncoder(graph, task)
    features = None
    fold_ids = np.empty(len(labels), dtype=np.int64)
    for fold, (fit, query) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(pairs.T, labels)):
        support = pairs[:, fit[labels[fit] == 1]]
        frame = encoder.transform(pairs[:, query], support)
        if features is None:
            features = np.zeros((len(labels), frame.shape[1]), dtype=np.float32)
            names = list(frame.columns)
        features[query] = frame.values
        fold_ids[query] = fold
    evaluation = encoder.transform(evaluation_pairs, pairs[:, labels == 1]).values
    return features, evaluation, names, fold_ids


def fit_transform(train, evaluation):
    transformer = QuantileTransformer(n_quantiles=min(256, len(train)),
        output_distribution='normal', subsample=100000, random_state=20260910)
    train_scaled = transformer.fit_transform(train).astype(np.float32)
    evaluation_scaled = transformer.transform(evaluation).astype(np.float32)
    return train_scaled, evaluation_scaled, transformer


class BatchEnsembleLinear(nn.Module):
    def __init__(self, input_dim, output_dim, members, first=False):
        super().__init__()
        bound = 1 / math.sqrt(input_dim)
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim).uniform_(-bound, bound))
        self.r = nn.Parameter(torch.ones(members, input_dim))
        self.s = nn.Parameter(torch.ones(members, output_dim))
        self.bias = nn.Parameter(torch.empty(output_dim).uniform_(-bound, bound).repeat(members, 1))
        if first:
            with torch.no_grad():
                self.r.bernoulli_(.5).mul_(2).sub_(1)

    def forward(self, x):
        return ((x*self.r) @ self.weight.T)*self.s + self.bias


class GraphTabM(nn.Module):
    def __init__(self, input_dim, width=128, depth=3, members=16, dropout=.1):
        super().__init__()
        self.members = members
        self.blocks = nn.ModuleList([BatchEnsembleLinear(input_dim if i == 0 else width,
            width, members, first=i == 0) for i in range(depth)])
        self.dropout = nn.Dropout(dropout)
        self.output_weight = nn.Parameter(torch.empty(members, width).uniform_(-1/math.sqrt(width), 1/math.sqrt(width)))
        self.output_bias = nn.Parameter(torch.zeros(members))

    def forward(self, x):
        h = x[:, None].expand(-1, self.members, -1)
        for layer in self.blocks:
            h = self.dropout(torch.relu(layer(h)))
        return (h*self.output_weight).sum(-1) + self.output_bias


class PairMLP(nn.Module):
    def __init__(self, input_dim, width=128, depth=3, dropout=.1):
        super().__init__()
        layers = []
        for i in range(depth):
            layers += [nn.Linear(input_dim if i == 0 else width, width), nn.ReLU(), nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, x):
        return self.network(x)
