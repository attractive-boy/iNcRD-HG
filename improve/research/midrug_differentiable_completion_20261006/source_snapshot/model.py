"""Same query-repaired forward function, gradients through completion weights."""
import torch
from common import HERE,read_module

PREVIOUS=HERE.parent/'midrug_graph_completion_20261006'
previous=read_module('stopped_graph_completion',PREVIOUS/'model.py')
CONFIGS={name:dict(mode='soft',reconstruction_weight=weight,auxiliary_weight=.1,completion_gradient=True)
         for name,weight in [('full',0.),('full_rec10',10.)]}

# Keep the already checked expression byte-for-byte: unwrap only no_grad.
query_fields=previous.query_fields.__wrapped__
first_states=previous.first_states
rectangle_sum=previous.rectangle_sum
completion_fields=previous.completion_fields
mask_generator=previous.mask_generator
auxiliary_loss=previous.auxiliary_loss

class DifferentiablePaths(previous.SoftPaths):
    def features(self,encoder,ps,dense,r,d):
        hm,hd,qr,qd=first_states(encoder,dense,r,d,ps.A);s=self.scorer
        mr,dr,qr,qd=s.mi(hm),s.drug(hd),s.mi(qr),s.drug(qd)
        direct=s.pairs(qr,qd)
        P=s.matrix(mr,dr).sigmoid()
        row=s.matrix(qr,dr).sigmoid();col=s.matrix(mr,qd).t().sigmoid()
        fields=query_fields(ps.A,ps.known,P,row,col,r,d,'soft')
        return direct,fields

def configure(a,data,cfg):
    a=previous.configure(a,data,cfg)
    if cfg.get('completion_gradient',False):a.model.soft_paths.__class__=DifferentiablePaths
    return a

def make_adapter(data,name,device):
    if name not in CONFIGS:return previous.make_adapter(data,name,device)
    original_name='soft_rec10' if CONFIGS[name]['reconstruction_weight'] else 'soft'
    a=previous.make_adapter(data,original_name,device)
    a.model.soft_paths.__class__=DifferentiablePaths
    return a
