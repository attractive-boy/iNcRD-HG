"""Two-block updates with matched forwards and a single BN-statistic update."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from common import Adapter,make_graph,Reconstruction

CONFIGS={name:dict(order=name,reconstruction_weight=10.) for name in ['consumer_first','representation_first','joint']}
def is_consumer(name):return name.startswith(('lin1.','lin2.','lin3.'))
def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name!='control':a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    return a
def mask_generator(device,seed):
    g=torch.Generator(device=device);g.manual_seed(seed);return g
def make_optimizers(m):
    named=list(m.named_parameters())
    c=[p for n,p in named if is_consumer(n)]
    r=[p for n,p in named if not is_consumer(n) and not n.startswith('reconstruction.')]
    rec=list(m.reconstruction.parameters())
    assert not (set(map(id,c))&set(map(id,r+rec)))
    assert set(map(id,c+r+rec))==set(map(id,m.parameters()))
    return dict(C=torch.optim.RMSprop(c,lr=1e-4,alpha=.99),
        R=torch.optim.RMSprop([dict(params=r,lr=1e-4),dict(params=rec,lr=1e-4,weight_decay=1e-4)],alpha=.99))
def clear(opts):
    for o in opts.values():o.zero_grad(set_to_none=True)
def update_blocks(parameters,opts,closure,order):
    """Closure(pass_index) does forward, returns a differentiable scalar and stats."""
    assert order in CONFIGS;clear(opts)
    first,first_stats=closure(0);assert torch.isfinite(first);first.backward()
    saved=None
    if order=='joint':saved=[None if p.grad is None else p.grad.detach().clone() for p in parameters]
    else:opts['C' if order=='consumer_first' else 'R'].step()
    clear(opts);second,second_stats=closure(1);assert torch.isfinite(second);second.backward()
    if order=='joint':
        for p,g in zip(parameters,saved):p.grad=g
        opts['C'].step();opts['R'].step()
    else:opts['R' if order=='consumer_first' else 'C'].step()
    return first_stats,second_stats
@contextmanager
def batch_statistics_without_tracking(m):
    assert m.training and m.eb.training and m.eb.track_running_stats
    old=m.eb.track_running_stats;m.eb.track_running_stats=False
    try:yield
    finally:m.eb.track_running_stats=old
def objective(a,indices,weight):
    z=a.logits('train');bce=F.binary_cross_entropy_with_logits(z,a.y)
    rec,_=a.model.reconstruction.loss(a.model.encoder,a.graph,indices,False)
    loss=bce+weight*rec
    return loss,dict(bce=float(bce.detach()),reconstruction=float(rec.detach()),objective=float(loss.detach()))
def train_epoch(a,opts,cfg,epoch,masks):
    m=a.model;m.train();old_count=int(m.eb.num_batches_tracked)
    indices=m.reconstruction.masks(a.graph,.5,masks);weight=10.*min(1.,epoch/50.)
    def closure(pass_index):
        if pass_index==0:return objective(a,indices,weight)
        with batch_statistics_without_tracking(m):return objective(a,indices,weight)
    first,second=update_blocks(list(m.parameters()),opts,closure,cfg['order'])
    assert int(m.eb.num_batches_tracked)==old_count+1 and m.eb.track_running_stats
    return dict(first_bce=first['bce'],first_reconstruction=first['reconstruction'],first_objective=first['objective'],
        second_bce=second['bce'],second_reconstruction=second['reconstruction'],second_objective=second['objective'],
        reconstruction_weight=weight,first_update_bce_change=second['bce']-first['bce'])
