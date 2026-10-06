import torch
from torch import Tensor



def log_prob(x: Tensor, logit: Tensor, t: Tensor) -> Tensor:
    log_numerator = logit + torch.log(t) + (t - 1) * (torch.log(x) + torch.log1p(-x))
    log_denominator = 2 * torch.logaddexp(logit + t * torch.log1p(-x), t * torch.log(x))
    log_prob_exact = log_numerator - log_denominator
    log_eps = torch.log(torch.tensor(1e-6, device=x.device))
    fn = torch.logaddexp(log_prob_exact, log_eps)
    return fn



def cdf(x: Tensor, logit: Tensor, t: Tensor) -> Tensor:
    log_term = (t * torch.log(x) - t * torch.log1p(-x) - logit)
    fn = torch.sigmoid(log_term)
    return fn



def icdf(u: Tensor, logit: Tensor, t: Tensor) -> Tensor:
    log_term = (logit + torch.log(u) - torch.log1p(-u)) / t
    fn = torch.sigmoid(log_term)
    return fn
