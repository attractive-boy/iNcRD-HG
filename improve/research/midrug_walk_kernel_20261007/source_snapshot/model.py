"""A typed product-graph walk branch on immutable query-deleted subgraphs."""
import ast,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import HERE,CLEAN,Adapter,make_graph,Reconstruction,base,sha256,read_module
OLD=HERE.parent/'midrug_subgraph_20261006'
EXTRACT_SOURCE=OLD/'extract.py'
Extractor=read_module('walk_kernel_extractor',EXTRACT_SOURCE).Extractor
SOURCE_PLAN=HERE.parent/'midrug_training_size_20261007/results/plan.json'
CONFIGS={name:dict(mode=name,k=16,hops=1,steps=3,prototypes=8,prototype_nodes=4,hidden=16) for name in ['pathwise','endpoints','fixed','no_edges','mpnn']}

# Reuse the exact historical local GNN without importing its cache or CLI.
tree=ast.parse((OLD/'model.py').read_text());ns=dict(nn=nn,torch=torch,F=F)
exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name in ['LocalLayer','SubgraphGNN']],type_ignores=[]),str(OLD/'model.py'),'exec'),ns)
SubgraphGNN=ns['SubgraphGNN']

class GraphCache:
    def __init__(self,data,device):
        self.device=device;self.ext=Extractor(data);self.arrays=None;fold=data['inner_fold']
        original=base.load_inner(fold)
        if np.array_equal(data['train_pairs'],original['train_pairs']) and np.array_equal(data['train_y'],original['train_y']) and np.array_equal(data['val_pairs'],original['val_pairs']):
            p=OLD/'cache'/f'f{fold}_h1_k16.npz';r=json.loads(p.with_suffix('.json').read_text())
            assert r['source_sha256']==sha256(EXTRACT_SOURCE) and r['cache_sha256']==sha256(p)
            assert r['inner_data_sha256']==sha256(CLEAN/'inner_data.npz') and r['clean_graph_sha256']==sha256(CLEAN/'graph.pt')
            with np.load(p) as z:
                self.arrays={key:torch.as_tensor(np.concatenate([z['train_'+key],z['val_'+key]]),device=device,
                    dtype=torch.long if key in ['rid','did'] else torch.float32) for key in ['rid','did','features','pos']}
            pairs=np.concatenate([data['train_pairs'],data['val_pairs']]);self.lookup=torch.full((605,216),-1,device=device,dtype=torch.long)
            pp=torch.as_tensor(pairs,device=device);self.lookup[pp[:,0],pp[:,1]]=torch.arange(len(pp),device=device)

    def select(self,r,d):
        if self.arrays is not None:
            ix=self.lookup[r,d]
            if (ix>=0).all():return {key:value[ix] for key,value in self.arrays.items()}
        rows=self.ext.many(torch.stack([r,d],1).cpu().numpy(),16,1)
        return {key:torch.as_tensor(rows[key],device=self.device,dtype=torch.long if key in ['rid','did'] else torch.float32) for key in ['rid','did','features','pos']}

def sym_normalize(a):
    inv=a.sum(-1).clamp_min(1).rsqrt()
    return a*inv.unsqueeze(-1)*inv.unsqueeze(-2)

def kernel_values(s,a,h,mode):
    """s[B,G,H,N], symmetric a[B,N,N], h[G,H,H]; p=0,1,2,3."""
    state=s;values=[s.square().sum((-1,-2))]
    for step in range(3):
        if mode=='no_edges':message=state
        else:message=torch.einsum('ghi,bgin->bghn',h,torch.matmul(state,a[:,None]))
        if mode=='endpoints':state=message;value=(s*state).sum((-1,-2))
        else:state=s*message;value=state.sum((-1,-2))
        values.append(value)
    return torch.stack(values,-1)

class WalkKernel(nn.Module):
    def __init__(self,xr_dim,xd_dim,mode):
        super().__init__();self.mode=mode;self.role=nn.Linear(20,16)
        self.rproj=nn.Sequential(nn.LayerNorm(xr_dim),nn.Linear(xr_dim,16))
        self.dproj=nn.Sequential(nn.LayerNorm(xd_dim),nn.Linear(xd_dim,16))
        self.norm=nn.LayerNorm(16)
        self.prototypes=nn.Parameter(torch.randn(8,4,16)*.25)
        self.edge_raw=nn.Parameter(torch.randn(8,2,2)*.25,requires_grad=mode not in ['fixed','no_edges'])
        self.head=nn.Sequential(nn.Linear(32,64),nn.LeakyReLU(.1),nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def hidden_graph(self):
        w=F.softplus(self.edge_raw);zero=torch.zeros_like(w)
        return sym_normalize(torch.cat([torch.cat([zero,w],-1),torch.cat([w.transpose(-1,-2),zero],-1)],-2))

    def features(self,batch,xr,xd):
        rid,did=batch['rid'],batch['did'];mr,md=rid<605,did<216;mask=torch.cat([mr,md],1)
        r,d=self.rproj(xr),self.dproj(xd)
        r=torch.cat([r,torch.zeros_like(r[:1])]);d=torch.cat([d,torch.zeros_like(d[:1])])
        x=self.norm(self.role(batch['features'])+torch.cat([r[rid],d[did]],1))*mask[:,:,None]
        s=torch.sigmoid(torch.einsum('bnf,ghf->bghn',x,self.prototypes)/4)
        types=torch.zeros((4,32),dtype=torch.bool,device=x.device);types[:2,:16]=True;types[2:,16:]=True
        s=s*mask[:,None,None,:]*types[None,None]
        pos=batch['pos'];zero=torch.zeros_like(pos)
        a=sym_normalize(torch.cat([torch.cat([zero,pos],-1),torch.cat([pos.transpose(-1,-2),zero],-1)],-2))
        value=kernel_values(s,a,self.hidden_graph(),self.mode)
        # Same positive p=0 channel and size denominator across every kernel arm.
        value=value/(4*mask.sum(1).clamp_min(1))[:,None,None]
        return torch.log1p(value).flatten(1)

    def forward(self,batch,xr,xd):return self.head(self.features(batch,xr,xd)).view(-1)

class KernelModel(base.Model):
    def forward(self,eli,ps,qis,ease,dense):
        batch=self.cache.select(*eli)
        if self.kernel_config=='mpnn':batch={**batch,'neg':torch.zeros_like(batch['pos'])}
        return super().forward(eli,ps,qis,ease,dense)+self.subgraph(batch,dense[0],dense[1])

def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name=='control':return a
    a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    if name=='unmasked10':return a
    m=a.model;m.__class__=KernelModel;m.kernel_config=name;m.cache=GraphCache(data,device)
    if name=='mpnn':m.subgraph=SubgraphGNN(dict(k=16,hidden=32,layers=3,covariates=True,signed=False),a.graph['miRNA'].x.shape[1],a.graph['drug'].x.shape[1]).to(device)
    else:m.subgraph=WalkKernel(a.graph['miRNA'].x.shape[1],a.graph['drug'].x.shape[1],name).to(device)
    for n,p in m.named_parameters():p.requires_grad_(n.startswith('subgraph.') and (n!='subgraph.edge_raw' or name not in ['fixed','no_edges']))
    return a

def source_record(fold):
    row=json.loads(SOURCE_PLAN.read_text())['controls']['unmasked10'][str(fold)];p=Path(row['path'])
    assert sha256(p)==row['metadata_sha256'];r=json.loads(p.read_text())
    assert sha256(p.with_suffix('.pt'))==r['checkpoint_sha256']==row['checkpoint_sha256']
    assert sha256(p.with_name(p.stem+'_inner.npz'))==row['probability_sha256']
    return p,r

def initialize(data,name,fold,device):
    a=make_adapter(data,name,device);p,r=source_record(fold)
    loaded=a.model.load_state_dict(torch.load(p.with_suffix('.pt'),map_location=device,weights_only=True),strict=False)
    assert not loaded.unexpected_keys and set(loaded.missing_keys)=={k for k in a.model.state_dict() if k.startswith('subgraph.')}
    return a,p,r

def make_optimizer(m):return torch.optim.RMSprop([p for p in m.parameters() if p.requires_grad],lr=1e-4,alpha=.99,weight_decay=1e-4)
