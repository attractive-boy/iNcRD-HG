"""Training-only manifold interpolation with a fixed same-drug comparison."""
import numpy as np
import torch
from torch.nn import functional as F
from common import Adapter,make_graph,Reconstruction

CONFIGS={name:dict(partner=mode,mixture_weight=.5,reconstruction_weight=10.) for name,mode in [('drug_mix','drug'),('global_mix','global')]}

def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name!='control':a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    a.mix_pairs=np.asarray(data['train_pairs']);return a

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

def mixing_plan(pairs,mode,fold,epoch):
    assert mode in ['drug','global']
    rng=np.random.default_rng(91007000+10000*fold+epoch);lam=float(rng.uniform())
    drug=pairs[:,1];groups=[np.flatnonzero(drug==d) for d in np.unique(drug)]
    groups=[idx for idx in groups if len(idx)>1]
    eligible=np.sort(np.concatenate(groups)) if groups else np.empty(0,dtype=np.int64)
    pi=np.arange(len(pairs))
    if mode=='global':groups=[eligible] if len(eligible)>1 else []
    for indices in groups:
        order=rng.permutation(indices);shift=int(rng.integers(1,len(order)));pi[order]=np.roll(order,shift)
    return pi,lam,eligible

def interpolate(h,y,pi,lam):
    partner=torch.as_tensor(pi,device=h.device,dtype=torch.long)
    return lam*h+(1-lam)*h[partner],lam*y+(1-lam)*y[partner]

def association_loss(a,cfg,epoch,fold,lam_override=None,identity=False):
    z,h=capture_features(a);clean=F.binary_cross_entropy_with_logits(z,a.y)
    pi,lam,eligible=mixing_plan(a.mix_pairs,cfg['partner'],fold,epoch)
    if lam_override is not None:lam=lam_override
    if identity:pi=np.arange(len(pi))
    mixed_h,mixed_y=interpolate(h,a.y,pi,lam)
    mixed=F.binary_cross_entropy_with_logits(decode(a.model,mixed_h),mixed_y)
    alpha=cfg['mixture_weight']*min(1.,max(0.,(epoch-30)/30.))
    loss=(1-alpha)*clean+alpha*mixed
    stats=dict(association_loss=float(loss.detach()),clean_bce=float(clean.detach()),mixed_bce=float(mixed.detach()),alpha=alpha,
        mixing_lambda=lam,eligible_count=len(eligible),same_drug_fraction=float(np.mean(a.mix_pairs[:,1]==a.mix_pairs[pi,1])),
        label_mean_error=float(abs(mixed_y.mean()-a.y.mean())),label_disagreement_fraction=float(np.mean(a.y.detach().cpu().numpy()!=a.y.detach().cpu().numpy()[pi])))
    return loss,stats,dict(clean=clean,mixed=mixed,features=h,mixed_features=mixed_h,mixed_labels=mixed_y,logits=z,partner=pi)
