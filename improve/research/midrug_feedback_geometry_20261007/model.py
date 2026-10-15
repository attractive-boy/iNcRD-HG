"""New feedback branch with all selected original M123 parameters/buffers frozen."""
import json
from pathlib import Path
import torch
from torch import nn
from common import HERE,Adapter,make_graph,Reconstruction,base,sha256
from modules import GeometryFeedback
CONFIGS={n:dict(moment=n,freeze_main=True,blocks=4,block_size=4,recurrent_steps=2) for n in ['full','diagonal','mean']}
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
    if name=='unmasked10':return a
    m=a.model;m.__class__=FeedbackModel
    m.feedback=GeometryFeedback(a.graph['miRNA'].x.shape[1],a.graph['drug'].x.shape[1],name).to(device)
    m.feedback.prepare_graph(a.graph)
    m.feedback_projection=nn.Linear(2*m.feedback.hidden,m.lin1.out_features,bias=False).to(device)
    nn.init.zeros_(m.feedback_projection.weight)
    for n,p in m.named_parameters():p.requires_grad_(n.startswith(('feedback.','feedback_projection.')))
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
    return torch.optim.RMSprop([p for p in m.parameters() if p.requires_grad],lr=1e-4,alpha=.99,weight_decay=1e-4)
