import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor

from probability import RelaxedBernoulli
from probability.LowRankGaussianCopula import LowRankGaussianCopula

from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import PairTensor, OptTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax 

from typing import Optional, Union



# ----------------------------------------------------------------------------------------------- #



class CopulaModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.eta: float = args.model.eta
        self.num_samples: int = args.model.num_samples
        self.woodbury: bool = args.model.woodbury
        self.marginal = RelaxedBernoulli

    def log_prob(self, *inputs) -> Tensor:
        return self.marginal.log_prob(*inputs)

    def cdf(self, *inputs) -> Tensor:
        return self.marginal.cdf(*inputs)
    
    def icdf(self, *inputs) -> Tensor:
        return self.marginal.icdf(*inputs)
    
    def loss_fn(self, logit, t, corr_factor, corr_diag, G: Data) -> float:
        trn_mask = G.uni_trn_mask

        # Label Smoothing.
        trn_label = G.uni_edge_weight[trn_mask].clamp(min=self.eta, max=1-self.eta)

        # Marginal Distribution.
        trn_logit = logit[trn_mask]
        trn_t = t[trn_mask]

        # Inter-edge Correlation.
        trn_corr_factor = corr_factor[trn_mask]
        trn_corr_diag = corr_diag[trn_mask]

        # Joint Distribution via LowRankGaussianCopula.
        copula = LowRankGaussianCopula(corr_factor=trn_corr_factor, corr_diag=trn_corr_diag, woodbury=self.woodbury)

        # Negative Log-Likelihood.
        u = self.cdf(trn_label, trn_logit, trn_t)
        nll_copula = -1 * copula.log_prob(u)
        log_pdf = self.log_prob(trn_label, trn_logit, trn_t)
        nll_marginal = -1 * log_pdf.sum()

        return (nll_copula + nll_marginal) / trn_label.size(0)

    def predict(self, G: Data, uni_trn_mask: Tensor, uni_eval_mask: Tensor):
        logit, t, corr_factor, corr_diag = self.forward(G)

        # Joint Distribution via LowRankGaussianCopula.
        copula = LowRankGaussianCopula(corr_factor=corr_factor, corr_diag=corr_diag, woodbury=self.woodbury)

        trn_label = G.uni_edge_weight[uni_trn_mask].clamp(min=self.eta, max=1-self.eta)
        
        # Marginal Distribution.
        trn_logit = logit[uni_trn_mask]
        trn_t = t[uni_trn_mask]
        eval_logit = logit[uni_eval_mask]
        eval_t = t[uni_eval_mask]

        # Prediction with Conditional Sampling
        cond_u = self.cdf(trn_label, trn_logit, trn_t).clamp(min=self.eta, max=1-self.eta)
        cond_idx = torch.where(uni_trn_mask)[0]
        sample_idx = torch.where(uni_eval_mask)[0]

        eval_u = copula.conditional_sample(
            cond_val=cond_u,
            sample_shape=[self.num_samples, ],
            cond_idx=cond_idx,
            sample_idx=sample_idx
        ).clamp(self.eta, 1 - self.eta)

        pred_y = self.icdf(eval_u, eval_logit, eval_t).mean(dim=0)
        return pred_y



# ----------------------------------------------------------------------------------------------- #



class CopulaLSP(CopulaModel):
    def __init__(self, args):
        super(CopulaLSP, self).__init__(args)

        self.eps: float = args.model.eps

        self.input_dim : int = args.model.embed_dim
        self.output_dim: int = args.model.embed_dim
        self.num_layers: int = args.model.num_layers

        self.input_conv = SNEAConv(
            self.input_dim, 
            self.output_dim // 2, 
            first_aggr=True, 
        )
        self.activation = nn.Tanh()
        self.convs = nn.ModuleList( 
            [
                SNEAConv(self.output_dim // 2, self.output_dim // 2, first_aggr=False) 
                for _ in range(self.num_layers - 1) 
            ]
        )
        self.weight = nn.Linear(self.output_dim, self.output_dim)
        self.linear_projection = nn.Linear(self.output_dim, 2)
        self.reset_parameters()

    def reset_parameters(self):
        self.input_conv.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.weight.reset_parameters()
        self.linear_projection.reset_parameters()

    def forward(self, G: Data) -> Tensor:
        x = G.x
        edge_index = G.edge_index[:, G.trn_mask] 
        edge_weight = G.edge_weight[G.trn_mask]

        pos_edge_index = edge_index[:, edge_weight > 0]
        neg_edge_index = edge_index[:, edge_weight < 0]

        x = self.input_conv(x, pos_edge_index, neg_edge_index)
        x = self.activation(x)

        for conv in self.convs:
            x = conv(x, pos_edge_index, neg_edge_index)
            x = self.activation(x)

        x = self.weight(x)
        x = self.activation(x)

        v_i = x[G.uni_edge_index[0, :]]  # node embeddings for source nodes
        v_j = x[G.uni_edge_index[1, :]]  # node embeddings for target nodes

        Q = v_i * v_j  # element-wise product to get edge embeddings

        logit, t = self.linear_projection(Q).unbind(dim=-1)
        t = F.sigmoid(t)

        # Efficient Gaussian Copula Operations via LowRankMultivariateNormal
        diag_cov_efficient = torch.sum(Q * Q, dim=1) + self.eps
        std_efficient = torch.sqrt(diag_cov_efficient)
        P = Q / std_efficient.unsqueeze(1)
        K = self.eps / diag_cov_efficient
        return logit, t, P, K



# ----------------------------------------------------------------------------------------------- #



class SNEAConv(MessagePassing):
    def __init__(self, input_dim: int, output_dim: int, first_aggr: bool, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.first_aggr = first_aggr

        self.lin_b = torch.nn.Linear(input_dim, output_dim, bias)
        self.lin_u = torch.nn.Linear(input_dim, output_dim, bias)

        self.alpha_u = torch.nn.Linear(self.output_dim * 2, 1)
        self.alpha_b = torch.nn.Linear(self.output_dim * 2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_b.reset_parameters()
        self.lin_u.reset_parameters()
        torch.nn.init.xavier_normal_(self.alpha_b.weight)
        torch.nn.init.xavier_normal_(self.alpha_u.weight)

    def forward(self, x: Union[Tensor, PairTensor], pos_edge_index: LongTensor, neg_edge_index: LongTensor):
        if self.first_aggr:
            h_b = self.lin_b(x)
            h_u = self.lin_u(x)

            edge, _ = remove_self_loops(pos_edge_index)
            edge, _ = add_self_loops(edge)
            edge_p = torch.zeros(edge.size(-1), dtype=torch.long)

            x1 = h_b
            x2 = h_b
            out_b = self.propagate(edge, x1=x1, x2=x2, edge_p=edge_p, alpha_func=self.alpha_b)

            edge, _ = remove_self_loops(neg_edge_index)
            edge, _ = add_self_loops(edge)
            edge_p = torch.zeros(edge.size(-1), dtype=torch.long)
            x1 = h_u
            x2 = h_u
            out_u = self.propagate(edge, x1=x1, x2=x2, edge_p=edge_p, alpha_func=self.alpha_u)
            out = torch.cat([out_b, out_u], dim=-1)

        else:
            F_in = self.input_dim
            h_b = x[..., :F_in]
            h_u = x[..., F_in:]

            edge1, _ = remove_self_loops(pos_edge_index)
            edge1, _ = add_self_loops(edge1)
            edge2, _ = remove_self_loops(neg_edge_index)
            edge = torch.cat([edge1, edge2], dim=-1)
            edge_p1 = torch.zeros(edge1.size(-1), dtype=torch.long)
            edge_p2 = torch.ones(edge2.size(-1), dtype=torch.long)
            edge_p = torch.cat([edge_p1, edge_p2], dim=-1)
            x1 = self.lin_b(h_b)
            x2 = self.lin_b(h_u)
            out_b = self.propagate(edge, x1=x1, x2=x2, edge_p=edge_p, alpha_func=self.alpha_b)

            edge1, _ = remove_self_loops(pos_edge_index)
            edge1, _ = add_self_loops(edge1)
            edge2, _ = remove_self_loops(neg_edge_index)
            edge = torch.cat([edge1, edge2], dim=-1)
            edge_p1 = torch.zeros(edge1.size(-1), dtype=torch.long)
            edge_p2 = torch.ones(edge2.size(-1), dtype=torch.long)
            edge_p = torch.cat([edge_p1, edge_p2], dim=-1)
            x1 = self.lin_u(h_u)
            x2 = self.lin_u(h_b)
            out_u = self.propagate(edge, x1=x1, x2=x2, edge_p=edge_p, alpha_func=self.alpha_u)

            out = torch.cat([out_b, out_u], dim=-1)
        return out

    def message(self, x1_j: Tensor, x2_j: Tensor, x1_i: Tensor, x2_i: Tensor, edge_p: Tensor, alpha_func, index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        x1 = torch.cat([x1_j, x1_i], dim=-1)
        x2 = torch.cat([x2_j, x2_i], dim=-1)
        edge_h = torch.stack([x1, x2], dim=-1)
        edge_h = edge_h[torch.arange(edge_h.size(0)), :, edge_p]
        alpha = alpha_func(edge_h)
        alpha = torch.tanh(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        x_i = torch.stack([x1_i, x2_i], dim=-1)
        x_i = x_i[torch.arange(edge_h.size(0)), :, edge_p]
        return x_i * alpha
