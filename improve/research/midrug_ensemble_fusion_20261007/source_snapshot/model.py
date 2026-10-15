"""Joint ensemble fusion after the original exact M1/M2/M3 representations."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from common import Adapter, make_graph, Reconstruction, base

CONFIGS = {
    'independent_rec10': dict(members=16, objective='member_bce', reconstruction_weight=10.),
    'mixture_rec10': dict(members=16, objective='mixture_bce', reconstruction_weight=10.),
}


class EnsembleLinear(nn.Module):
    def __init__(self, linear, members, first=False):
        super().__init__()
        self.weight = linear.weight
        self.r = nn.Parameter(linear.weight.new_ones(members, linear.in_features))
        self.s = nn.Parameter(linear.weight.new_ones(members, linear.out_features))
        self.bias = nn.Parameter(linear.bias.detach().repeat(members, 1))
        if first and members > 1:
            with torch.no_grad():
                self.r.bernoulli_(.5).mul_(2).sub_(1)

    def forward(self, x):
        return F.linear(x*self.r, self.weight)*self.s+self.bias


class EnsembleOutput(nn.Module):
    def __init__(self, linear, members):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.detach().repeat(members, 1))
        self.bias = nn.Parameter(linear.bias.detach().repeat(members))
        if members > 1:
            bound = 1/math.sqrt(linear.in_features)
            with torch.no_grad():
                self.weight[1:].uniform_(-bound, bound)
                self.bias[1:].uniform_(-bound, bound)

    def forward(self, x):
        return (x*self.weight).sum(-1)+self.bias


def mixture_logit(logits):
    """Stable log-odds of the mean of member probabilities, without clipping."""
    return torch.logsumexp(F.logsigmoid(logits), -1)-torch.logsumexp(F.logsigmoid(-logits), -1)


def classification_loss(logits, labels, objective):
    if objective == 'member_bce':
        return F.binary_cross_entropy_with_logits(logits, labels[:, None].expand_as(logits))
    if objective == 'mixture_bce':
        return F.binary_cross_entropy_with_logits(mixture_logit(logits), labels)
    raise ValueError(objective)


class EnsembleModel(base.Model):
    def fusion_features(self, eli, ps, qis, ease, dense):
        # This is exactly the original Model.forward through its concatenation.
        r, d = eli
        xr, xd, Sr1, Sd1, A = dense
        zr, zd, zm, zdd, corr = self.encoder.forward_dense(xr, xd, Sr1, Sd1, A, r, d, ps.qvals(r, d))
        z = {'miRNA': zr, 'drug': zd}
        parts = [zm, zdd]
        parts += [lin(e) for lin, e in zip(self.ncn_lin, ps.qncn(r, d, z, corr))]
        parts.append(self.pc(qis))
        parts.append(self.em(self.eb(ease)))
        return torch.cat(parts, -1)

    def head_logits(self, features):
        h = features[:, None, :].expand(-1, self.members, -1)
        h = F.leaky_relu(self.lin1(h), .1)
        h = F.leaky_relu(self.lin2(h), .1)
        return self.lin3(h)

    def member_logits(self, eli, ps, qis, ease, dense):
        return self.head_logits(self.fusion_features(eli, ps, qis, ease, dense))

    def forward(self, eli, ps, qis, ease, dense):
        return mixture_logit(self.member_logits(eli, ps, qis, ease, dense))


def make_adapter(data, name, device):
    a = Adapter(data, make_graph(data), device)
    if name == 'control':
        return a
    if name == 'unmasked10' or name.endswith('_rec10'):
        a.model.add_module('reconstruction', Reconstruction(a.graph, a.cfg.hidden).to(device))
    if name == 'unmasked10':
        return a
    members = 1 if name.startswith('identity') else CONFIGS[name if name.endswith('_rec10') else name+'_rec10']['members']
    m = a.model
    m.__class__ = EnsembleModel
    m.members = members
    m.lin1 = EnsembleLinear(m.lin1, members, first=True)
    m.lin2 = EnsembleLinear(m.lin2, members)
    m.lin3 = EnsembleOutput(m.lin3, members)
    return a


def member_logits(a, part):
    return a.model.member_logits(a.pairs[part], a.ps, a.qis[part], a.ease[part], a.dense)


def make_optimizer(m):
    groups = [dict(params=[p for n, p in m.named_parameters() if not n.startswith('reconstruction.')], lr=1e-4)]
    if hasattr(m, 'reconstruction'):
        groups.append(dict(params=m.reconstruction.parameters(), lr=1e-4, weight_decay=1e-4))
    return torch.optim.RMSprop(groups, alpha=.99)


def mask_generator(device, seed):
    return torch.Generator(device=device).manual_seed(seed)
