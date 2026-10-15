"""Supervised last-block Q/V LoRA of the existing original ChemBERTa.

The fixed first-five-layer states contain only original drug SMILES. The
normalized learned change is anchored to the unchanged original drug input.
"""
import json,math
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from transformers import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaLayer
from common import HERE,Adapter,make_graph,Reconstruction,sha256
from incrd_hg_m123.encoder import QueryHollowEncoder

CONFIGS={n+s:dict(mode=n,reconstruction_weight=w,rank=8,adaptation_lr=1e-4)
         for n in ['token','pooled'] for s,w in [('',0.),('_rec10',10.)]}

def row_normalize(x):
    return (x-x.amin(1,keepdim=True))/(x.amax(1,keepdim=True)-x.amin(1,keepdim=True)+1e-8)

class LoRALinear(nn.Module):
    def __init__(self,base,rank):
        super().__init__();self.base=base
        self.down=nn.Linear(base.in_features,rank,bias=False)
        self.up=nn.Linear(rank,base.out_features,bias=False);nn.init.zeros_(self.up.weight)
    def forward(self,x):return self.base(x)+self.up(self.down(x))

class DrugAdaptation(nn.Module):
    def __init__(self,cfg,device):
        super().__init__();self.mode=cfg['mode'];self.rank=cfg['rank']
        meta=json.loads((HERE/'input_manifest.json').read_text())
        assert sha256(HERE/'token_cache.pt')==meta['cache_sha256']
        assert sha256(HERE/'original_last_layer.pt')==meta['last_layer_sha256']
        cache=torch.load(HERE/'token_cache.pt',map_location='cpu',weights_only=True)
        config=RobertaConfig.from_json_file(str(HERE.parents[3]/'.external/models/hf_home/hub/models--seyonec--ChemBERTa-zinc-base-v1/snapshots'/meta['revision']/'config.json'))
        config._attn_implementation='eager'
        self.block=RobertaLayer(config)
        self.block.load_state_dict(torch.load(HERE/'original_last_layer.pt',map_location='cpu',weights_only=True))
        self.block.requires_grad_(False);self.block.to(device).eval()
        # Trim trailing masked positions per length bucket. Real-token states do
        # not attend to padding, and positions are already encoded in the cache.
        lengths=cache['attention_mask'].sum(1);self.groups=[]
        for limit in [32,64,128,256,512]:
            ix=torch.nonzero((lengths<=limit)&(lengths>(0 if limit==32 else limit//2)),as_tuple=False).flatten()
            for start in range(0,len(ix),8):
                ids=ix[start:start+8];width=int(lengths[ids].max())
                self.groups.append((ids.to(device),cache['last_input'][ids,:width].to(device),cache['attention_mask'][ids,:width].to(device)))
        assert sum(len(g[0]) for g in self.groups)==216
        with torch.no_grad():reference=self.token_pool()
        torch.testing.assert_close(reference.cpu(),cache['mean_masked'],atol=2e-5,rtol=2e-5)
        self.register_buffer('reference',reference,persistent=False)
        self.register_buffer('normalized_reference',row_normalize(reference),persistent=False)
        if self.mode=='token':
            self.block.attention.self.query=LoRALinear(self.block.attention.self.query,self.rank).to(device)
            self.block.attention.self.value=LoRALinear(self.block.attention.self.value,self.rank).to(device)
        else:
            # Same trainable parameter count as two rank-r Q/V updates.
            self.down=nn.Linear(768,2*self.rank,bias=False).to(device)
            self.up=nn.Linear(2*self.rank,768,bias=False).to(device);nn.init.zeros_(self.up.weight)
        self.train(False)

    def train(self,mode=True):
        super().train(mode)
        # Deterministic attribute transform for query deletion and batch invariance.
        self.block.eval()
        return self

    def token_pool(self):
        ids,rows=[],[]
        for ix,h,mask in self.groups:
            extended=(1-mask[:,None,None,:].to(h.dtype))*torch.finfo(h.dtype).min
            z=self.block(h,attention_mask=extended)[0]
            rows.append((z*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]);ids.append(ix)
        indices=torch.cat(ids);values=torch.cat(rows)
        return values[torch.argsort(indices)]

    def forward(self,x):
        if self.mode=='token':pooled=self.token_pool()
        else:pooled=self.reference+self.up(F.gelu(self.down(self.reference)))
        return x+(row_normalize(pooled)-self.normalized_reference)

    def adaptation_parameters(self):return [p for p in self.parameters() if p.requires_grad]

class AdaptiveEncoder(QueryHollowEncoder):
    @contextmanager
    def feature_context(self):
        if getattr(self,'_adapted_drug',None) is not None:
            yield
            return
        self._adapted_drug=self.drug_adaptation(self.graph['drug'].x)
        try:yield
        finally:self._adapted_drug=None

    def layer(self,convs,x):
        if convs is self.original.conv1:
            with self.feature_context():return super().layer(convs,{**x,'drug':self._adapted_drug})
        return super().layer(convs,x)

    def forward_dense(self,xr,xd,Sr1,Sd1,A,r=None,d=None,q=None):
        with self.feature_context():return super().forward_dense(xr,self._adapted_drug,Sr1,Sd1,A,r,d,q)

    def forward(self,x,edges):
        with self.feature_context():return self.original({**x,'drug':self._adapted_drug},edges)

def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name=='control':return a
    if name=='unmasked10' or CONFIGS[name]['reconstruction_weight']:
        a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    if name=='unmasked10':return a
    a.model.encoder.__class__=AdaptiveEncoder
    a.model.encoder.drug_adaptation=DrugAdaptation(CONFIGS[name],device)
    a.model.encoder._adapted_drug=None
    return a

def mask_generator(device,seed):return torch.Generator(device=device).manual_seed(seed)

def make_optimizers(m,cfg):
    groups=[dict(params=[p for n,p in m.named_parameters() if not n.startswith(('encoder.drug_adaptation.','reconstruction.'))],lr=1e-4)]
    if cfg['reconstruction_weight']:groups.append(dict(params=m.reconstruction.parameters(),lr=1e-4,weight_decay=1e-4))
    original=torch.optim.RMSprop(groups,alpha=.99)
    adaptation=torch.optim.AdamW(m.encoder.drug_adaptation.adaptation_parameters(),lr=cfg['adaptation_lr'],weight_decay=.01)
    return original,adaptation
