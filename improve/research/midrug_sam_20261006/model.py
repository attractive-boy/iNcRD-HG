"""Same M123 inference; exact two-pass SAM or matched random weight perturbation."""
import torch
from torch.nn import functional as F
from common import Adapter,make_graph,Reconstruction

CONFIGS={n+suffix:dict(direction=n,rho=.05,reconstruction_weight=weight)
         for n in ['sam','random'] for suffix,weight in [('',0.),('_rec10',10.)]}

def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name=='unmasked10' or CONFIGS.get(name,{}).get('reconstruction_weight',0):
        a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    return a

def generator(device,seed):
    g=torch.Generator(device=device);g.manual_seed(seed);return g

mask_generator=generator

def make_optimizer(model):
    groups=[dict(params=[p for n,p in model.named_parameters() if not n.startswith('reconstruction.')],lr=1e-4)]
    if hasattr(model,'reconstruction'):
        groups.append(dict(params=list(model.reconstruction.parameters()),lr=1e-4,weight_decay=1e-4))
    return torch.optim.RMSprop(groups,alpha=.99)

def closure_for(a,indices,weight):
    def closure():
        bce=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)
        rec=bce.new_zeros(())
        if indices is not None:rec,_=a.model.reconstruction.loss(a.model.encoder,a.graph,indices,False)
        loss=bce+weight*rec
        return loss,dict(loss=float(loss.detach()),bce=float(bce.detach()),reconstruction=float(rec.detach()))
    return closure

def perturbation_step(model,opt,closure,rho,direction,rng=None,inspect=None):
    """One optimizer update, two backprops; restore exact weights and second-pass buffers.

    Both losses use the SAME reconstruction indices, chosen outside closure. The
    gradient through the ascent direction is deliberately stopped (first-order SAM).
    RMSprop's weight decay is applied at restored weights, outside the ascent loss.
    """
    if direction not in ['sam','random']:raise ValueError(direction)
    if rho<0:raise ValueError(rho)
    opt.zero_grad();loss,first=closure()
    if not torch.isfinite(loss):raise FloatingPointError('first loss')
    loss.backward()
    eligible=[(n,p) for n,p in model.named_parameters() if p.grad is not None]
    if not eligible:raise RuntimeError('No gradients')
    if not all(torch.isfinite(p.grad).all() for _,p in eligible):raise FloatingPointError('first gradient')
    gnorm=torch.stack([p.grad.norm() for _,p in eligible]).norm()
    originals={n:p.detach().clone() for n,p in eligible}
    # BatchNorm in training still uses batch statistics in pass two. Restore its
    # updated buffers afterwards, retaining exactly ONE running-statistics update.
    buffers={n:b.detach().clone() for n,b in model.named_buffers()}
    with torch.no_grad():
        if direction=='sam':vectors=[p.grad.detach().clone() for _,p in eligible]
        else:
            if rng is None:raise ValueError('Random control needs a separate generator')
            vectors=[torch.randn(p.shape,device=p.device,dtype=p.dtype,generator=rng) for _,p in eligible]
        norm=torch.stack([v.norm() for v in vectors]).norm()
        deltas=[v*(rho/(norm+1e-12)) for v in vectors]
        perturb_norm=float(torch.stack([v.norm() for v in deltas]).norm())
        alignment=float(sum((p.grad*d).sum() for (_,p),d in zip(eligible,deltas))/(gnorm*max(perturb_norm,1e-30)+1e-30))
    try:
        with torch.no_grad():
            for (n,p),delta in zip(eligible,deltas):p.copy_(originals[n]+delta)
        if inspect is not None:inspect(model,originals,dict(zip([n for n,_ in eligible],deltas)))
        opt.zero_grad();loss2,second=closure()
        if not torch.isfinite(loss2):raise FloatingPointError('perturbed loss')
        loss2.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('second gradient')
    finally:
        with torch.no_grad():
            for n,p in eligible:p.copy_(originals[n])
            for n,b in model.named_buffers():b.copy_(buffers[n])
    opt.step()
    return dict(first=first,second=second,gradient_norm=float(gnorm),perturbation_norm=perturb_norm,alignment=alignment)
