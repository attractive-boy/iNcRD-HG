"""Per-query M1/M2 direction and scale inputs, preserving original graph channels."""
import torch
from torch import nn
from torch.nn import functional as F
from common import Adapter,make_graph,Reconstruction,base

CONFIGS={'direction_scale':{},'direction':{},'raw':{}}


def decompose(h):
    mean=h.mean(-1,keepdim=True)
    scale=torch.sqrt((h-mean).square().mean(-1,keepdim=True)+1e-5)
    direction=(h-mean)/scale
    # asinh is invertible and smooth at zero; log1p preserves the positive scale.
    stats=torch.cat([torch.asinh(mean),torch.log1p(scale)],-1)
    return direction,stats


def combine(parts,mode):
    if mode not in CONFIGS:raise ValueError(mode)
    outputs=[];statistics=[]
    for part in parts[:-2]:
        direction,stats=decompose(part)
        outputs.append(part if mode=='raw' else direction)
        statistics.append(stats if mode=='direction_scale' else torch.zeros_like(stats))
    return torch.cat(outputs+parts[-2:]+statistics,-1)


class NormalizedFusionModel(base.Model):
    def parts(self,eli,ps,qis,ease,dense):
        r,d=eli
        zr,zd,zm,zdd,corr=self.encoder.forward_dense(*dense,r,d,ps.qvals(r,d))
        z={'miRNA':zr,'drug':zd}
        result=[zm,zdd]
        result += [lin(e) for lin,e in zip(self.ncn_lin,ps.qncn(r,d,z,corr))]
        result.append(self.pc(qis));result.append(self.em(self.eb(ease)))
        return result

    def forward(self,eli,ps,qis,ease,dense):
        h=combine(self.parts(eli,ps,qis,ease,dense),self.fusion_mode)
        h=F.leaky_relu(self.lin1(h),.1);h=F.leaky_relu(self.lin2(h),.1)
        return self.lin3(h).view(-1)


def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name!='control':a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    if name in CONFIGS:
        m=a.model;m.__class__=NormalizedFusionModel;m.fusion_mode=name
        n=2*(2+len(m.ncn_lin));m.extra_width=n
        weight=m.lin1.weight
        m.lin1.weight=nn.Parameter(torch.cat([weight.detach().clone(),weight.new_zeros(weight.shape[0],n)],1))
        m.lin1.in_features+=n
    return a


def mask_generator(device,seed):return torch.Generator(device=device).manual_seed(seed)


def make_optimizer(m):
    return torch.optim.RMSprop([
        dict(params=[p for n,p in m.named_parameters() if not n.startswith('reconstruction.')],lr=1e-4),
        dict(params=m.reconstruction.parameters(),lr=1e-4,weight_decay=1e-4)],alpha=.99)


def step(a,opt,epoch,generator):
    m=a.model;m.train();opt.zero_grad(set_to_none=True)
    logits=a.logits('train');bce=F.binary_cross_entropy_with_logits(logits,a.y)
    indices=m.reconstruction.masks(a.graph,.5,generator);auxiliary,_=m.reconstruction.loss(m.encoder,a.graph,indices,False)
    aw=10*min(1.,epoch/50.);loss=bce+aw*auxiliary
    assert torch.isfinite(loss);loss.backward();opt.step();m.eval()
    return dict(bce=float(bce.detach()),reconstruction=float(auxiliary.detach()),reconstruction_weight=aw,loss=float(loss.detach()))
