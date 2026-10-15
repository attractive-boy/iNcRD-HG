"""Fixed encoders/buffers, trainable original fusion head and optional feedback."""
import json
from pathlib import Path
import torch
from torch import nn
from common import HERE,Adapter,make_graph,Reconstruction,base,sha256
from modules import GeometryFeedback
CONFIGS={n:dict(moment=moment,train_head=True,freeze_encoder=True,freeze_buffers=True) for n,moment in [('full_head','full'),('mean_head','mean'),('head_only',None)]}
HEAD_PREFIXES=('lin1.','lin2.','lin3.')
BRANCH_PREFIXES=('feedback.','feedback_projection.')
def trainable(name):return name.startswith(HEAD_PREFIXES+BRANCH_PREFIXES)
SOURCE_PLAN=HERE.parent/'midrug_training_size_20261007/results/plan.json'

class FeedbackModel(base.Model):
    def forward(self,eli,ps,qis,ease,dense):
        h=self.feedback(dense[0],dense[1],eli)
        addition=self.feedback_projection(h)
        hook=self.lin1.register_forward_hook(lambda module,args,out:out+addition)
        try:return super().forward(eli,ps,qis,ease,dense)
        finally:hook.remove()

def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name=='control':return a
    a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    if name in ['unmasked10','head_only']:
        if name=='head_only':
            for n,p in a.model.named_parameters():p.requires_grad_(n.startswith(HEAD_PREFIXES))
        return a
    m=a.model;m.__class__=FeedbackModel
    m.feedback=GeometryFeedback(a.graph['miRNA'].x.shape[1],a.graph['drug'].x.shape[1],CONFIGS[name]['moment']).to(device)
    m.feedback.prepare_graph(a.graph)
    m.feedback_projection=nn.Linear(2*m.feedback.hidden,m.lin1.out_features,bias=False).to(device)
    nn.init.zeros_(m.feedback_projection.weight)
    for n,p in m.named_parameters():p.requires_grad_(trainable(n))
    a.freeze_main=True
    return a

def source_record(fold):
    r=json.loads(SOURCE_PLAN.read_text())['controls']['unmasked10'][str(fold)]
    path=Path(r['path']);assert sha256(path)==r['metadata_sha256']
    record=json.loads(path.read_text());assert sha256(path.with_suffix('.pt'))==record['checkpoint_sha256']==r['checkpoint_sha256']
    assert sha256(path.with_name(path.stem+'_inner.npz'))==r['probability_sha256']
    return path,record

def initialize(data,name,fold,device):
    a=make_adapter(data,name,device);path,r=source_record(fold)
    loaded=a.model.load_state_dict(torch.load(path.with_suffix('.pt'),map_location=device,weights_only=True),strict=False)
    assert not loaded.unexpected_keys
    assert set(loaded.missing_keys)=={k for k in a.model.state_dict() if k.startswith(('feedback.','feedback_projection.'))}
    return a,path,r

def make_optimizer(m):
    head=[p for n,p in m.named_parameters() if n.startswith(HEAD_PREFIXES) and p.requires_grad]
    branch=[p for n,p in m.named_parameters() if n.startswith(BRANCH_PREFIXES) and p.requires_grad]
    groups=[dict(params=head,lr=1e-5,weight_decay=0.)]
    if branch:groups.append(dict(params=branch,lr=1e-4,weight_decay=1e-4))
    return torch.optim.RMSprop(groups,alpha=.99)
