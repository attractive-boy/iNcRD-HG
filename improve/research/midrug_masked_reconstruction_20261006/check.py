"""Scope, query deletion, no auxiliary-forward mutation and gradient checks."""
import json,time
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,load_inner,seed_all
from model import make_adapter,mask_generator
import check_modules

def grad_stats(loss,params):
    gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,gs)])

def main():
    data=load_inner(0);seed_all(42);base=make_adapter(data,'control','cuda:0');base.model.eval()
    seed_all(42);a=make_adapter(data,'masked01','cuda:0');a.model.eval();rec=a.model.reconstruction
    assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in k.lower() for k in a.model.state_dict())
    with torch.no_grad():p0,p1=base.predict('val'),a.predict('val')
    for name,value in base.model.state_dict().items():torch.testing.assert_close(value,a.model.state_dict()[name],atol=0,rtol=0)
    np.testing.assert_allclose(p0,p1,atol=1e-6,rtol=1e-5)
    original={name:a.graph[name].x.clone() for name in a.graph.node_types}
    indices=rec.masks(a.graph,.5,mask_generator('cuda:0',123))
    loss,by_type=rec.loss(a.model.encoder,a.graph,indices,True)
    for name,x in original.items():torch.testing.assert_close(a.graph[name].x,x,atol=0,rtol=0)
    with torch.no_grad():p2=a.predict('val')
    np.testing.assert_allclose(p1,p2,atol=1e-6,rtol=1e-5)
    # Original inner-held-out labels cannot change any input, mask target or reconstruction loss.
    changed={**data,'val_y':1-data['val_y']};seed_all(42);b=make_adapter(changed,'masked01','cuda:0');b.model.eval()
    other,_=b.model.reconstruction.loss(b.model.encoder,b.graph,indices,True)
    torch.testing.assert_close(loss,other,atol=1e-6,rtol=1e-5)
    params=list(a.model.encoder.parameters());gb=grad_stats(F.binary_cross_entropy_with_logits(a.logits('train'),a.y),params);gr=grad_stats(loss,params)
    assert torch.isfinite(gb).all() and torch.isfinite(gr).all() and gr.norm()>0
    gradients=dict(bce_norm=float(gb.norm()),reconstruction_norm=float(gr.norm()),cosine=float(F.cosine_similarity(gb[None],gr[None])))
    a.model.train();opt=torch.optim.RMSprop(a.model.parameters(),lr=1e-4);opt.zero_grad()
    (F.binary_cross_entropy_with_logits(a.logits('train'),a.y)+.1*loss).backward()
    assert rec.token['miRNA'].grad.abs().max()>0 and rec.decoder['drug'][-1].weight.grad.abs().max()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.model.parameters())
    opt.step();checks=check_modules.check(a,count=6)
    r=dict(status='passed',initial_control_weights_exact=True,initial_control_max_prediction_error=float(abs(p0-p1).max()),
        auxiliary_forward_prediction_max_error=float(abs(p1-p2).max()),auxiliary_forward_does_not_mutate_inputs=True,
        inner_validation_label_invariance=True,finite_nonzero_auxiliary_gradients=True,gradient_diagnostics=gradients,
        reconstruction_initial={k:float(v) for k,v in by_type.items()},full_logit_deletion=checks,
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'initial_checks.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)

if __name__=='__main__':main()
