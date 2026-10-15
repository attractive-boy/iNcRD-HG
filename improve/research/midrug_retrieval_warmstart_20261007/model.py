"""Same zero-residual retrieval branch on frozen versus trainable warm starts."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from common import HERE,PREVIOUS,read_module,sha256
prior=read_module('audited_label_retrieval',PREVIOUS/'model.py')
mask_generator=prior.mask_generator
CONFIGS={'frozen_label':dict(freeze_main=True,branch='label'),
         'joint_label':dict(freeze_main=False,branch='label'),
         'frozen_features':dict(freeze_main=True,branch='features')}
SOURCE_PLAN=HERE.parent/'midrug_training_size_20261007/results/plan.json'

def make_adapter(data,name,device):
    cfg=CONFIGS.get(name);a=prior.make_adapter(data,cfg['branch'] if cfg else name,device)
    a.freeze_main=cfg['freeze_main'] if cfg else False
    if a.freeze_main:
        for n,p in a.model.named_parameters():p.requires_grad_(n.startswith('case_memory.'))
    return a

def source_record(fold):
    reference=json.loads(SOURCE_PLAN.read_text())['controls']['unmasked10'][str(fold)]
    path=Path(reference['path']);assert sha256(path)==reference['metadata_sha256']
    record=json.loads(path.read_text());assert sha256(path.with_suffix('.pt'))==record['checkpoint_sha256']==reference['checkpoint_sha256']
    assert sha256(path.with_name(path.stem+'_inner.npz'))==reference['probability_sha256']
    return path,record

def initialize(data,name,fold,device):
    a=make_adapter(data,name,device);path,record=source_record(fold)
    result=a.model.load_state_dict(torch.load(path.with_suffix('.pt'),map_location=device,weights_only=True),strict=False)
    assert not result.unexpected_keys
    assert set(result.missing_keys)=={k for k in a.model.state_dict() if k.startswith('case_memory.')}
    return a,path,record

def make_optimizer(model):
    case=[p for n,p in model.named_parameters() if n.startswith('case_memory.') and p.requires_grad]
    main=[p for n,p in model.named_parameters() if not n.startswith(('case_memory.','reconstruction.')) and p.requires_grad]
    rec=[p for n,p in model.named_parameters() if n.startswith('reconstruction.') and p.requires_grad]
    groups=[dict(params=case,lr=1e-4,weight_decay=1e-4)]
    if main:groups.append(dict(params=main,lr=1e-5,weight_decay=0.))
    if rec:groups.append(dict(params=rec,lr=1e-5,weight_decay=1e-4))
    return torch.optim.RMSprop(groups,alpha=.99)

def resumed_sampler(a,fold,source_epoch):
    generator=mask_generator(a.y.device,20261006+fold)
    for _ in range(source_epoch):a.model.reconstruction.masks(a.graph,.5,generator)
    return generator

def step(a,opt,name,epoch,generator):
    m=a.model
    if a.freeze_main:m.eval();m.case_memory.train()
    else:m.train()
    opt.zero_grad(set_to_none=True);bce=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)
    indices=m.reconstruction.masks(a.graph,.5,generator)
    if a.freeze_main:
        with torch.no_grad():rec,_=m.reconstruction.loss(m.encoder,a.graph,indices,False)
    else:rec,_=m.reconstruction.loss(m.encoder,a.graph,indices,False)
    loss=bce+10.*rec;assert torch.isfinite(loss);loss.backward();opt.step()
    return dict(bce=float(bce.detach()),reconstruction=float(rec.detach()),weight=10.,loss=float(loss.detach()),main_frozen=a.freeze_main)
