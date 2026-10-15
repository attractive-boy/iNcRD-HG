"""VICReg-style single-view regularization of query-corrected M1 endpoints."""
import torch
from torch.nn import functional as F
from common import Adapter,make_graph,Reconstruction

CONFIGS={'varcov':dict(var=.2,cov=.2),'var':dict(var=.2,cov=0.),'cov':dict(var=0.,cov=.2)}


def regularizers(h):
    # Per-query feature normalization is used only in this training objective.
    z=F.layer_norm(h,(h.shape[1],),eps=1e-5)
    centered=z-z.mean(0,keepdim=True)
    covariance=centered.T@centered/(len(z)-1)
    variance=F.relu(1-torch.sqrt(covariance.diag()+1e-4)).mean()
    offdiag=covariance[~torch.eye(h.shape[1],device=h.device,dtype=torch.bool)]
    redundancy=offdiag.square().sum()/h.shape[1]
    return variance,redundancy


def training_forward(a):
    cache={}
    def capture(module,args):cache['h']=args[0]
    handle=a.model.lin1.register_forward_pre_hook(capture)
    try:logits=a.logits('train')
    finally:handle.remove()
    width=a.cfg.hidden;endpoints=(cache['h'][:,:width],cache['h'][:,width:2*width])
    values=[regularizers(h) for h in endpoints]
    return logits,.5*(values[0][0]+values[1][0]),.5*(values[0][1]+values[1][1])


def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name!='control':a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    if name in CONFIGS:a.regularizer_config=CONFIGS[name]
    return a


def mask_generator(device,seed):return torch.Generator(device=device).manual_seed(seed)


def make_optimizer(m):
    return torch.optim.RMSprop([
        dict(params=[p for n,p in m.named_parameters() if not n.startswith('reconstruction.')],lr=1e-4),
        dict(params=m.reconstruction.parameters(),lr=1e-4,weight_decay=1e-4)],alpha=.99)


def step(a,opt,epoch,generator):
    m=a.model;m.train();opt.zero_grad(set_to_none=True)
    logits,var,cov=training_forward(a);bce=F.binary_cross_entropy_with_logits(logits,a.y)
    indices=m.reconstruction.masks(a.graph,.5,generator);auxiliary,_=m.reconstruction.loss(m.encoder,a.graph,indices,False)
    ramp=min(1.,max(0.,(epoch-30)/30.));vw=a.regularizer_config['var']*ramp;cw=a.regularizer_config['cov']*ramp
    aw=10*min(1.,epoch/50.)
    loss=((bce+vw*var)+cw*cov)+aw*auxiliary
    assert torch.isfinite(loss);loss.backward();opt.step();m.eval()
    return dict(bce=float(bce.detach()),variance=float(var.detach()),covariance=float(cov.detach()),
                variance_weight=vw,covariance_weight=cw,reconstruction=float(auxiliary.detach()),
                reconstruction_weight=aw,loss=float(loss.detach()))
