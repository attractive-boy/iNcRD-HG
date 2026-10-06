# -*- coding: utf-8 -*-
"""Faithful PyTorch port of the official HGNNLDA (TCBB 2023) training model.

Source of truth: repos/HGNNLDA-official/main.py (TensorFlow 1.15).  Notes on details
that were replicated verbatim, including quirks:

  * Message is ALWAYS A @ E0 (the initial embedding variable), not the previous
    layer's embedding.  Official line: `new = sparse_matmul(H_u, self.lncRNA_embeddings)`.
  * Layer update: E = leaky_relu(A @ E0 @ W_i + E_prev), dropout(0.1) when training,
    then L2-normalize rows.
  * Final embedding = concat([E0, E1, E2, E3]) (E0 is the raw, un-normalized variable).
  * Score = <u_final, v_final> (inner product).
  * BPR pairwise loss with per-positive negative sampling.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hypergraph_adjacency(A):
    """A: (n_l, n_d) dense/scipy bipartite adjacency of TRAIN positives.

    Returns (A_u, A_i) as float32 torch tensors:
        A_u = D_l^{-1/2} H  D_e^{-1} H^T D_l^{-1/2}   (n_l x n_l)
        A_i = D_d^{-1/2} H^T D_l^{-1} H  D_d^{-1/2}   (n_d x n_d)
    matching main.py initModel().
    """
    A = np.asarray(A, dtype=np.float64)
    dl = A.sum(axis=1)          # lncRNA degrees
    dd = A.sum(axis=0)          # drug degrees
    inv_sqrt_dl = np.where(dl > 0, 1.0 / np.sqrt(np.maximum(dl, 1e-12)), 0.0)
    inv_sqrt_dd = np.where(dd > 0, 1.0 / np.sqrt(np.maximum(dd, 1e-12)), 0.0)
    inv_dl = np.where(dl > 0, 1.0 / np.maximum(dl, 1e-12), 0.0)
    inv_dd = np.where(dd > 0, 1.0 / np.maximum(dd, 1e-12), 0.0)

    # temp1[i,j] = A[i,j] * sqrt(1/dl[i]);  temp2 = temp1^T
    temp1 = A * inv_sqrt_dl[:, None]
    temp2 = temp1.T
    A_u = (temp1 * inv_dd[None, :]) @ temp2

    B = A.T
    t1 = B * inv_sqrt_dd[:, None]
    t2 = t1.T
    A_i = (t1 * inv_dl[None, :]) @ t2
    return (torch.tensor(A_u, dtype=torch.float32),
            torch.tensor(A_i, dtype=torch.float32))


class HGNNLDA(nn.Module):
    def __init__(self, n_lncrna, n_drug, emb_size=64, n_layer=3, dropout=0.1, stddev=0.005):
        super().__init__()
        self.n_layer = n_layer
        self.dropout = dropout
        self.emb_lncrna = nn.Parameter(torch.randn(n_lncrna, emb_size) * stddev)
        self.emb_drug = nn.Parameter(torch.randn(n_drug, emb_size) * stddev)
        self.weights = nn.ParameterList()
        for _ in range(n_layer):
            w = torch.empty(emb_size, emb_size)
            nn.init.xavier_uniform_(w)
            self.weights.append(nn.Parameter(w))

    def forward(self, A_u, A_i, training, propagate='e0'):
        e0_l, e0_d = self.emb_lncrna, self.emb_drug
        # 'e0'  : message is always A @ E0 (faithful to the official code)
        # 'iter': message is A @ E_prev (the paper's Eq.10-11, iterated propagation)
        e_l, e_d = e0_l, e0_d
        all_l, all_d = [e0_l], [e0_d]
        for i in range(self.n_layer):
            if propagate == 'iter':
                msg_l = A_u @ e_l
                msg_d = A_i @ e_d
            else:
                msg_l = A_u @ e0_l
                msg_d = A_i @ e0_d
            e_l = F.leaky_relu(msg_l @ self.weights[i] + e_l)
            e_d = F.leaky_relu(msg_d @ self.weights[i] + e_d)
            if training:
                e_l = F.dropout(e_l, p=self.dropout)
                e_d = F.dropout(e_d, p=self.dropout)
            e_l = F.normalize(e_l, p=2, dim=1)
            e_d = F.normalize(e_d, p=2, dim=1)
            all_l.append(e_l)
            all_d.append(e_d)
        return torch.cat(all_l, dim=1), torch.cat(all_d, dim=1)


def bpr_loss(final_l, final_d, u_idx, i_idx, j_idx, reg_u):
    u = final_l[u_idx]
    v = final_d[i_idx]
    vn = final_d[j_idx]
    y = (u * v).sum(1) - (u * vn).sum(1)
    loss = -F.logsigmoid(y).sum()
    reg = reg_u * (u.pow(2).sum() + v.pow(2).sum() + vn.pow(2).sum())
    return loss + reg


def weight_reg(model, reg_u):
    r = 0.0
    for w in model.weights:
        r = r + w.pow(2).sum()
    return reg_u * r
