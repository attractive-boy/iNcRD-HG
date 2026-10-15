"""Original M123 inference; matched weight-initialized reconstruction fine-tuning."""
import json
from pathlib import Path
import torch
from common import HERE,Adapter,make_graph,Reconstruction,sha256

CONFIGS={'keep':dict(schedule='keep10'),'off':dict(schedule='off'),'decay':dict(schedule='linear100')}
LR=1e-5
SOURCE_PLAN=HERE.parent/'midrug_training_size_20261007/results/plan.json'


def reconstruction_weight(name,stage_epoch):
    if stage_epoch<1:raise ValueError('Only optimizer steps have a reconstruction weight')
    if name=='keep':return 10.
    if name=='off':return 0.
    if name=='decay':return 10.*max(0.,1.-stage_epoch/100.)
    raise ValueError(name)


def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name!='control':a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    return a


def source_record(fold):
    reference=json.loads(SOURCE_PLAN.read_text())['controls']['unmasked10'][str(fold)]
    path=Path(reference['path']);assert sha256(path)==reference['metadata_sha256']
    record=json.loads(path.read_text());assert sha256(path.with_suffix('.pt'))==record['checkpoint_sha256']==reference['checkpoint_sha256']
    assert sha256(path.with_name(path.stem+'_inner.npz'))==reference['probability_sha256']
    return path,record


def initialize(data,name,fold,device):
    a=make_adapter(data,name,device);path,record=source_record(fold)
    a.model.load_state_dict(torch.load(path.with_suffix('.pt'),map_location=device,weights_only=True))
    return a,path,record


def make_optimizer(m):
    return torch.optim.RMSprop([
        dict(params=[p for n,p in m.named_parameters() if not n.startswith('reconstruction.')],lr=LR),
        dict(params=m.reconstruction.parameters(),lr=LR,weight_decay=1e-4)],alpha=.99)


def mask_generator(device,seed):
    return torch.Generator(device=device).manual_seed(seed)


def resumed_sampler(a,fold,source_epoch):
    generator=mask_generator(a.y.device,20261006+fold)
    for _ in range(source_epoch):a.model.reconstruction.masks(a.graph,.5,generator)
    return generator


def step(a,opt,name,epoch,generator):
    from torch.nn import functional as F
    m=a.model;m.train();opt.zero_grad()
    bce=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)
    indices=m.reconstruction.masks(a.graph,.5,generator)
    weight=reconstruction_weight(name,epoch)
    if weight:
        auxiliary,_=m.reconstruction.loss(m.encoder,a.graph,indices,False)
    else:
        # Measure reconstruction drift without updating its decoder at weight zero.
        with torch.no_grad():auxiliary,_=m.reconstruction.loss(m.encoder,a.graph,indices,False)
    loss=bce+weight*auxiliary;assert torch.isfinite(loss);loss.backward();opt.step()
    return dict(bce=float(bce.detach()),reconstruction=float(auxiliary.detach()),weight=weight,loss=float(loss.detach()))
