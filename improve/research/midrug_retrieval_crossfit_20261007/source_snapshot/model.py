"""Same retrieval branch and inference base; only fitting offsets change."""
import json
import numpy as np
import torch
from common import HERE,RETRIEVAL,read_module,sha256,mask_generator
prior=read_module('original_crossfit_retrieval_branch',RETRIEVAL/'model.py')
CONFIGS={'oof_label':dict(offset='oof',branch='label'),
         'insample_label':dict(offset='in_sample',branch='label'),
         'oof_features':dict(offset='oof',branch='features')}

def make_adapter(data,name,device):
    cfg=CONFIGS.get(name);a=prior.make_adapter(data,cfg['branch'] if cfg else name,device)
    if cfg:
        for n,p in a.model.named_parameters():p.requires_grad_(n.startswith('case_memory.'))
    return a

def initialize(data,name,fold,device):
    a=make_adapter(data,name,device);p=HERE/'bases'/f'base_f{fold}_k-1.json';r=json.loads(p.read_text())
    assert sha256(p.with_suffix('.pt'))==r['checkpoint_sha256']
    loaded=a.model.load_state_dict(torch.load(p.with_suffix('.pt'),map_location=device,weights_only=True),strict=False)
    assert not loaded.unexpected_keys and set(loaded.missing_keys)=={k for k in a.model.state_dict() if k.startswith('case_memory.')}
    path=HERE/'offsets'/f'fold{fold}.npz'
    with np.load(path) as z:
        np.testing.assert_array_equal(z['train_pairs'],data['train_pairs']);np.testing.assert_array_equal(z['train_indices'],data['train_indices'])
        offsets={k:torch.as_tensor(z[k],device=device) for k in ['oof','in_sample','val']}
    return a,p,r,offsets

def make_optimizer(m):
    return torch.optim.RMSprop([p for p in m.parameters() if p.requires_grad],lr=1e-4,alpha=.99,weight_decay=1e-4)
