"""Training-only adversarial perturbations of query-hollow pair representations."""
import math
import torch
from torch.nn import functional as F
from common import Adapter,make_graph,Reconstruction

CONFIGS={n+suffix:dict(direction=n,normalized_rms=.05,mixture_weight=.5,reconstruction_weight=w)
         for n in ['adv','random'] for suffix,w in [('',0.),('_rec10',10.)]}

def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name=='unmasked10' or CONFIGS.get(name,{}).get('reconstruction_weight',0):
        a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    return a

def mask_generator(device,seed):
    g=torch.Generator(device=device);g.manual_seed(seed);return g

def make_optimizer(m):
    groups=[dict(params=[p for n,p in m.named_parameters() if not n.startswith('reconstruction.')],lr=1e-4)]
    if hasattr(m,'reconstruction'):groups.append(dict(params=list(m.reconstruction.parameters()),lr=1e-4,weight_decay=1e-4))
    return torch.optim.RMSprop(groups,alpha=.99)

def capture_features(a,part='train'):
    """Capture the original concat through a temporary read-only hook, with gradients."""
    values=[]
    handle=a.model.lin1.register_forward_pre_hook(lambda module,inputs:values.append(inputs[0]))
    try:logits=a.logits(part)
    finally:handle.remove()
    assert len(values)==1
    return logits,values[0]

def decode(m,h):
    return m.lin3(F.leaky_relu(m.lin2(F.leaky_relu(m.lin1(h),.1)),.1)).view(-1)

@torch.no_grad()
def feature_delta(features,gradient,rms,direction,rng=None):
    """Linearized ascent in an ellipsoid with per-column TRAIN std scales.

    Define u=delta/scale and ||u_i||_2 <= rms*sqrt(width). The exact
    maximizer of the linearized per-example loss is u_i parallel to scale*g_i.
    These scales, directions and active masks are detached from the loss graph.
    """
    if direction not in ['adv','random'] or rms<0:raise ValueError((rms,direction))
    scale=features.detach().std(0,unbiased=False).clamp_min(1e-3)
    g=gradient.detach()*scale
    norms=g.norm(dim=1,keepdim=True);active=norms>1e-12
    if direction=='adv':raw=g
    else:
        if rng is None:raise ValueError('Independent random control generator required')
        raw=torch.randn(features.shape,device=features.device,dtype=features.dtype,generator=rng)
    raw_norm=raw.norm(dim=1,keepdim=True)
    unit=raw/raw_norm.clamp_min(1e-12)*active
    normalized=unit*(rms*math.sqrt(features.shape[1]));delta=scale*normalized
    return delta,dict(scale=scale,normalized=normalized,active=active.squeeze(1),scaled_gradient=g)

def association_loss(a,cfg,epoch,rng,rms_override=None):
    logits,h=capture_features(a)
    clean=F.binary_cross_entropy_with_logits(logits,a.y)
    gradient=torch.autograd.grad(clean,h,retain_graph=True)[0]
    rms=cfg['normalized_rms'] if rms_override is None else rms_override
    delta,info=feature_delta(h,gradient,rms,cfg['direction'],rng)
    perturbed=decode(a.model,h+delta)
    augmented=F.binary_cross_entropy_with_logits(perturbed,a.y)
    alpha=cfg['mixture_weight']*min(1.,max(0.,(epoch-30)/30.))
    loss=(1-alpha)*clean+alpha*augmented
    active=info['active'];nr=info['normalized'].norm(dim=1)/math.sqrt(h.shape[1])
    gn=info['scaled_gradient'].norm(dim=1);un=info['normalized'].norm(dim=1)
    alignment=(info['scaled_gradient']*info['normalized']).sum(1)/(gn*un).clamp_min(1e-30)
    stats=dict(association_loss=float(loss.detach()),clean_bce=float(clean.detach()),perturbed_bce=float(augmented.detach()),
        alpha=alpha,width=h.shape[1],active_queries=int(active.sum()),total_queries=len(h),
        normalized_rms_mean=float(nr[active].mean()) if active.any() else 0.,
        normalized_rms_max_error=float((nr[active]-rms).abs().max()) if active.any() else 0.,
        alignment_mean=float(alignment[active].mean()) if active.any() else 0.,
        scale_min=float(info['scale'].min()),scale_max=float(info['scale'].max()),
        detach_direction=not delta.requires_grad)
    return loss,stats,dict(features=h,delta=delta,info=info,clean=clean,augmented=augmented,logits=logits)
