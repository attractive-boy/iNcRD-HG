"""Shared M123 predictor plus small, fixed-route residual experts."""
import torch
from torch import nn
from torch.nn import functional as F
from common import Adapter, make_graph, Reconstruction, base

CONFIGS = {'degree': {}, 'random': {}, 'uniform': {}}
N_DRUGS = 216
PRIME = 2147483647


def pair_hash(r, d):
    return ((r * N_DRUGS + d) * 1103515245 + 20261007) % PRIME


def degree_group(ps, r, d, cuts):
    q = ps.qvals(r, d)
    dr, dd = ps.deg_r[r]-q, ps.deg_d[d]-q
    return (dr > cuts[0]).long()*2 + (dd > cuts[1]).long()


def route_groups(kind, ps, r, d, degree_cuts, random_cuts):
    if kind == 'degree': return degree_group(ps, r, d, degree_cuts)
    if kind == 'random': return torch.bucketize(pair_hash(r, d), random_cuts, right=False)
    raise ValueError(kind)


class RegimeModel(base.Model):
    def fusion_features(self, eli, ps, qis, ease, dense):
        r, d = eli
        zr, zd, zm, zdd, corr = self.encoder.forward_dense(*dense, r, d, ps.qvals(r,d))
        z = {'miRNA':zr, 'drug':zd}
        parts = [zm,zdd]
        parts += [lin(e) for lin,e in zip(self.ncn_lin, ps.qncn(r,d,z,corr))]
        parts.append(self.pc(qis)); parts.append(self.em(self.eb(ease)))
        return torch.cat(parts,-1)

    def gate(self, eli, ps):
        if self.routing == 'uniform':
            return self.degree_cuts.new_full((eli.shape[1],4), .25)
        r,d = eli
        group=route_groups(self.routing,ps,r,d,self.degree_cuts,self.random_cuts)
        return F.one_hot(group,num_classes=4).to(self.degree_cuts.dtype)

    def components(self, eli, ps, qis, ease, dense):
        x=self.fusion_features(eli,ps,qis,ease,dense)
        h=F.leaky_relu(self.lin1(x),.1);h=F.leaky_relu(self.lin2(h),.1)
        shared=self.lin3(h).view(-1)
        normalized=F.layer_norm(x,(x.shape[1],))
        experts=torch.stack([expert(normalized).view(-1) for expert in self.experts],-1)
        gate=self.gate(eli,ps)
        return shared,(experts*gate).sum(-1),experts,gate

    def forward(self, eli, ps, qis, ease, dense):
        shared,residual,_,_=self.components(eli,ps,qis,ease,dense)
        return shared+residual


def make_adapter(data, name, device):
    kind=name.replace('_plain','')
    a=Adapter(data,make_graph(data),device)
    if name=='control':return a
    if not name.endswith('_plain'):
        a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    if name=='unmasked10':return a
    assert kind in CONFIGS
    m=a.model;m.__class__=RegimeModel;m.routing=kind
    r,d=a.pairs['train'];q=a.ps.qvals(r,d)
    # torch.quantile uses the midpoint for even sample counts, matching NumPy median.
    cuts=torch.stack((torch.quantile(a.ps.deg_r[r]-q,.5),torch.quantile(a.ps.deg_d[d]-q,.5)))
    counts=torch.bincount(degree_group(a.ps,r,d,cuts),minlength=4)
    assert (counts>0).all()
    ordered=pair_hash(r,d).sort().values
    random_cuts=ordered[counts.cumsum(0)[:-1]-1]
    m.register_buffer('degree_cuts',cuts)
    m.register_buffer('random_cuts',random_cuts)
    width=m.lin1.in_features
    m.experts=nn.ModuleList([nn.Sequential(nn.Linear(width,64),nn.LeakyReLU(.1),nn.Linear(64,1)) for _ in range(4)]).to(device)
    for expert in m.experts:
        nn.init.zeros_(expert[-1].weight);nn.init.zeros_(expert[-1].bias)
    return a


def mask_generator(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


def make_optimizer(model):
    return torch.optim.RMSprop([
        dict(params=[p for n,p in model.named_parameters() if not n.startswith(('reconstruction.','experts.'))],lr=1e-4),
        dict(params=model.reconstruction.parameters(),lr=1e-4,weight_decay=1e-4),
        dict(params=model.experts.parameters(),lr=1e-3,weight_decay=1e-4)],alpha=.99)


def step(a, optimizer, epoch, generator):
    m=a.model;m.train();optimizer.zero_grad(set_to_none=True)
    bce=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)
    indices=m.reconstruction.masks(a.graph,.5,generator)
    auxiliary,_=m.reconstruction.loss(m.encoder,a.graph,indices,False)
    weight=10*min(1.,epoch/50.);loss=bce+weight*auxiliary
    assert torch.isfinite(loss);loss.backward();optimizer.step();m.eval()
    return dict(bce=float(bce.detach()),reconstruction=float(auxiliary.detach()),weight=weight,loss=float(loss.detach()))
