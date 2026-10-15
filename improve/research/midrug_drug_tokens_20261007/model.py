"""miRNA-conditioned frozen original ChemBERTa tokens; equal-parameter pooling controls."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from common import HERE,Adapter,make_graph,Reconstruction,base
CONFIGS={name:dict(mode=name,hidden=32,chunk_size=256,branch_lr=3e-4,reconstruction_weight=10.)
         for name in ['mirna','population','pooled']}
_CACHE=None
def tokens():
 global _CACHE
 if _CACHE is None:_CACHE=torch.load(HERE/'final_tokens.pt',map_location='cpu',weights_only=True)
 return _CACHE

class DrugTokens(nn.Module):
 def __init__(self,xdim,rdim,hidden=32,mode='mirna',chunk_size=256):
  super().__init__();self.hidden,self.mode,self.chunk_size=hidden,mode,chunk_size
  self.norm=nn.LayerNorm(xdim)
  self.key=nn.Linear(xdim,hidden,bias=False);self.value=nn.Linear(xdim,hidden,bias=False)
  self.query=nn.Sequential(nn.LayerNorm(rdim),nn.Linear(rdim,hidden),nn.LeakyReLU(.1),nn.Linear(hidden,hidden,bias=False))
  self.readout=nn.Sequential(nn.LayerNorm(2*hidden),nn.Linear(2*hidden,hidden),nn.LeakyReLU(.1))
 def encode(self,z,mask,xr):
  z=self.norm(z);k,v=self.key(z),self.value(z)
  w=mask[:,:,None].to(z.dtype);den=w.sum(1)
  return k,v,self.query(xr),(k*w).sum(1)/den,(v*w).sum(1)/den
 def attend(self,k,v,q,km,vm,mask,r,d):
  qr=q[r]
  qa=q.mean(0,keepdim=True).expand_as(qr) if self.mode=='population' else qr
  scores=torch.bmm(k[d],qa[:,:,None]).squeeze(-1)/math.sqrt(self.hidden)
  alpha=scores.masked_fill(~mask[d],-torch.inf).softmax(-1)
  if self.mode=='pooled':
   alpha=mask[d].to(v.dtype)/mask[d].sum(-1,keepdim=True)
  pooled=torch.bmm(alpha[:,None,:],v[d]).squeeze(1)
  # Same global bilinear interaction in all three configurations keeps Q/K active in pooled control.
  out=self.readout(torch.cat([pooled,qr*km[d]],-1))
  return out,alpha
 def forward(self,z,mask,xr,r,d):
  encoded=self.encode(z,mask,xr);pieces=[]
  for s in range(0,len(r),self.chunk_size):
   out,_=self.attend(*encoded,mask,r[s:s+self.chunk_size],d[s:s+self.chunk_size]);pieces.append(out)
  return torch.cat(pieces) if pieces else xr.new_empty((0,self.hidden))

class TokenModel(base.Model):
 def forward(self,eli,ps,qis,ease,dense):
  r,d=eli
  addition=self.token_projection(self.token_context(self.drug_tokens,self.drug_mask,dense[0],r,d))
  hook=self.lin1.register_forward_hook(lambda module,args,out:out+addition)
  try:return super().forward(eli,ps,qis,ease,dense)
  finally:hook.remove()

def make_adapter(data,name,device):
 a=Adapter(data,make_graph(data),device)
 if name=='control':return a
 a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
 if name=='unmasked10':return a
 cfg=CONFIGS[name];m=a.model;m.__class__=TokenModel
 # Preserve all original/reconstruction initial values before drawing branch parameters.
 m.token_context=DrugTokens(768,a.graph['miRNA'].x.shape[1],cfg['hidden'],cfg['mode'],cfg['chunk_size']).to(device)
 m.token_projection=nn.Linear(cfg['hidden'],m.lin1.out_features,bias=False).to(device);nn.init.zeros_(m.token_projection.weight)
 c=tokens();m.register_buffer('drug_tokens',c['tokens'].to(device),persistent=False)
 m.register_buffer('drug_mask',c['mask'].to(device),persistent=False)
 return a

def make_optimizer(m,cfg):
 groups=[dict(params=[p for n,p in m.named_parameters() if not n.startswith(('token_context.','token_projection.','reconstruction.'))],lr=1e-4),
         dict(params=list(m.token_context.parameters())+list(m.token_projection.parameters()),lr=cfg['branch_lr'],weight_decay=1e-4),
         dict(params=m.reconstruction.parameters(),lr=1e-4,weight_decay=1e-4)]
 return torch.optim.RMSprop(groups,alpha=.99,eps=1e-8,momentum=0,centered=False)
def mask_generator(device,seed):return torch.Generator(device=device).manual_seed(seed)
