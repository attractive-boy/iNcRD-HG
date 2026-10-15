"""Proper binary partial losses; unchanged M123 inference model."""
import torch
from torch.nn import functional as F
from common import Adapter,make_graph,Reconstruction

CONFIGS={name:dict(loss=name,reconstruction_weight=10.) for name in ['negative_tail','positive_tail','brier']}
def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name!='control':a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    return a
def mask_generator(device,seed):
    g=torch.Generator(device=device);g.manual_seed(seed);return g
def make_optimizer(m):
    return torch.optim.RMSprop([dict(params=[p for n,p in m.named_parameters() if not n.startswith('reconstruction.')],lr=1e-4),
        dict(params=list(m.reconstruction.parameters()),lr=1e-4,weight_decay=1e-4)],alpha=.99)
def capture_features(a,part='train'):
    values=[];handle=a.model.lin1.register_forward_pre_hook(lambda m,args:values.append(args[0]))
    try:z=a.logits(part)
    finally:handle.remove()
    assert len(values)==1;return z,values[0]
def decode(m,h):return m.lin3(F.leaky_relu(m.lin2(F.leaky_relu(m.lin1(h),.1)),.1)).view(-1)
def phi(x):
    t=x.clamp_max(.01)
    series=t.square()*(.5+t*(-1/6+t*(1/24+t*(-1/120+t/720))))
    return torch.where(x<.01,series,x+torch.expm1(-x))
def proper_loss(z,y,name):
    assert name in CONFIGS
    if name=='negative_tail':return 2*((1-y)*torch.sigmoid(z)+y*phi(torch.logaddexp(torch.zeros_like(z),-z)))
    if name=='positive_tail':return 2*(y*torch.sigmoid(-z)+(1-y)*phi(torch.logaddexp(torch.zeros_like(z),z)))
    return 2*(y*torch.sigmoid(-z).square()+(1-y)*torch.sigmoid(z).square())
def gradient_factor(z,name):
    if name=='negative_tail':return 2*torch.sigmoid(-z)
    if name=='positive_tail':return 2*torch.sigmoid(z)
    return 4*torch.sigmoid(z)*torch.sigmoid(-z)
def association_loss(a,cfg,epoch,fold):
    z=a.logits('train');clean=F.binary_cross_entropy_with_logits(z,a.y)
    shaped=proper_loss(z,a.y,cfg['loss']).mean();alpha=min(1.,max(0.,(epoch-30)/30.))
    loss=(1-alpha)*clean+alpha*shaped;factor=gradient_factor(z.detach(),cfg['loss'])
    stats=dict(association_loss=float(loss.detach()),clean_bce=float(clean.detach()),proper_loss=float(shaped.detach()),alpha=alpha,
        positive_mean_factor=float(factor[a.y==1].mean()),negative_mean_factor=float(factor[a.y==0].mean()))
    return loss,stats,dict(logits=z,clean=clean,shaped=shaped)
