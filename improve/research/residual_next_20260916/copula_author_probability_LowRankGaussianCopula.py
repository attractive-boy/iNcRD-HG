import torch
from torch.distributions import (
    Distribution,
    LowRankMultivariateNormal,
    MultivariateNormal,
    Normal,
    constraints,
)



class LowRankGaussianCopula(Distribution):
    
    support = constraints.interval(0.0, 1.0)

    has_rsample = True

    arg_constraints = {
        'corr_factor': constraints.real_vector,
        'corr_diag': constraints.positive,
    }

    def __init__(self, corr_factor, corr_diag, woodbury, validate_args=None):

        self.corr_factor = corr_factor
        self.corr_diag = corr_diag
        self.woodbury = woodbury

        batch_shape = self.corr_factor.shape[:-2]
        event_shape = self.corr_factor.shape[-2:-1]

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

        self.low_rank_multivariate_normal = LowRankMultivariateNormal(
            loc=torch.zeros(event_shape, device=self.corr_factor.device),
            cov_factor=self.corr_factor,
            cov_diag=self.corr_diag,
            validate_args=validate_args,
        )
        
        self.standard_normal = Normal(
            torch.tensor(0.0, device=self.corr_factor.device),
            torch.tensor(1.0, device=self.corr_factor.device),
        )

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value_x = self.standard_normal.icdf(value)
        log_prob_mvn = self.low_rank_multivariate_normal.log_prob(value_x)
        log_prob_copula = log_prob_mvn + 0.5 * value_x.pow(2).sum(-1)
        return log_prob_copula
        
    def woodbury_reformulation(self, cond_val, sample_shape=torch.Size([]), cond_idx=None, sample_idx=None) -> torch.Tensor:
        n = self.event_shape[-1]
        m = cond_val.shape[-1]
        rank = self.corr_factor.shape[-1]
        
        if cond_idx is None:
            cond_idx = torch.arange(n - m, n, device=cond_val.device)
        if sample_idx is None:
            sample_idx = torch.tensor(
                [i for i in range(n) if i not in set(cond_idx.tolist())],
                device=cond_val.device
            )

        P_trn  = self.corr_factor.index_select(dim=-2, index=cond_idx)
        P_eval = self.corr_factor.index_select(dim=-2, index=sample_idx)
        
        K_trn  = self.corr_diag.index_select(dim=-1, index=cond_idx)
        K_eval = self.corr_diag.index_select(dim=-1, index=sample_idx)
        K_trn_inv = 1.0 / K_trn

        cond_val_nscale = self.standard_normal.icdf(cond_val)

        S = torch.eye(rank, device=P_trn.device) + P_trn.transpose(-1, -2) @ (K_trn_inv.unsqueeze(-1) * P_trn)
        v = P_trn.transpose(-1, -2) @ (K_trn_inv * cond_val_nscale).unsqueeze(-1)
        
        solved_v = torch.linalg.solve(S, v).squeeze(-1)
        cond_mu = P_eval @ solved_v

        W = S.inverse() + 1e-4 * torch.eye(S.shape[-1], device=S.device)
        L = torch.linalg.cholesky(W)
        new_cov_factor = P_eval @ L
        
        cond_low_rank_normal = LowRankMultivariateNormal(
            loc=cond_mu,
            cov_factor=new_cov_factor,
            cov_diag=K_eval
        )
        
        samples_nscale = cond_low_rank_normal.rsample(sample_shape)
        samples_uscale = self.standard_normal.cdf(samples_nscale)

        return samples_uscale

    def standard_method(self, cond_val, sample_shape=torch.Size([]), cond_idx=None, sample_idx=None) -> torch.Tensor:
        m = cond_val.shape[-1]
        n = self.event_shape[-1]

        if cond_idx is None:
            cond_idx = torch.arange(n - m, n, device=cond_val.device)
        if sample_idx is None:
            sample_idx = torch.tensor(
                [i for i in range(n) if i not in set(cond_idx.tolist())],
                device=cond_val.device
            )
        
        P_eval = self.corr_factor.index_select(dim=-2, index=sample_idx)
        P_trn  = self.corr_factor.index_select(dim=-2, index=cond_idx)
        K_eval = self.corr_diag.index_select(dim=-1, index=sample_idx)
        K_trn  = self.corr_diag.index_select(dim=-1, index=cond_idx)

        R_00 = P_trn @ P_trn.transpose(-1, -2) + torch.diag_embed(K_trn)
        R_10 = P_eval @ P_trn.transpose(-1, -2)
        R_01 = R_10.transpose(-1, -2)

        cond_val_nscale = self.standard_normal.icdf(cond_val)
        reg_coeff = torch.linalg.solve(R_00, R_01)

        cond_mu = (reg_coeff.transpose(-1, -2) @ cond_val_nscale.unsqueeze(-1)).squeeze(-1)

        R_11 = P_eval @ P_eval.transpose(-1, -2) + torch.diag_embed(K_eval)
        tmp = R_10 @ reg_coeff
        
        tmp = (tmp + tmp.transpose(-1, -2)) * 0.5
        cond_sigma = R_11 - tmp

        cond_normal = MultivariateNormal(loc=cond_mu, covariance_matrix=cond_sigma)
        samples_nscale = cond_normal.rsample(sample_shape)
        samples_uscale = self.standard_normal.cdf(samples_nscale)

        return samples_uscale

    def conditional_sample(self, cond_val, sample_shape=torch.Size([]), cond_idx=None, sample_idx=None) -> torch.Tensor:
        if self.woodbury:
            return self.woodbury_reformulation(cond_val, sample_shape, cond_idx, sample_idx)
        else:
            return self.standard_method(cond_val, sample_shape, cond_idx, sample_idx)
