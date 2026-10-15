"""A drug-structure branch supplements unchanged M1/M2/M3 and feature reconstruction."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from common import Adapter,Model,Reconstruction,make_graph,load_molecules

CONFIGS={f'{mode}_wd{power}':dict(mode=mode,weight_decay=10.**(-power),hidden=64,layers=3,
                               dropout=.1,branch_aux_weight=.1,reconstruction_weight=10.)
         for power in (4,3) for mode in ('gnn','bag','chem')}


class ResidualLayer(nn.Module):
    def __init__(self,h,dropout):
        super().__init__();self.eps=nn.Parameter(torch.zeros(()))
        self.mlp=nn.Sequential(nn.Linear(h,2*h),nn.GELU(),nn.Linear(2*h,h))
        self.norm=nn.LayerNorm(h);self.dropout=nn.Dropout(dropout)
    def forward(self,h,message):
        return self.norm(h+self.dropout(self.mlp((1+self.eps)*h+message)))


class DrugEncoder(nn.Module):
    def __init__(self,xd_width,molecules,metadata,cfg):
        super().__init__();self.mode=cfg['mode'];h=cfg['hidden'];self.ndrug=216
        for name,value in molecules.items():self.register_buffer(name,value)
        self.register_buffer('atom_counts',(molecules['atom_offsets'][1:]-molecules['atom_offsets'][:-1]).float())
        if self.mode=='chem':
            self.input=nn.Sequential(nn.LayerNorm(xd_width),nn.Linear(xd_width,h),nn.GELU())
        else:
            self.atom_embeddings=nn.ModuleList([nn.Embedding(n,h) for n in metadata['atom_cardinalities']])
            self.bond_embeddings=nn.ModuleList([nn.Embedding(n,h) for n in metadata['bond_cardinalities']])
        self.layers=nn.ModuleList([ResidualLayer(h,cfg['dropout']) for _ in range(cfg['layers'])])
        self.output=nn.Sequential(nn.Linear(h+1,h),nn.LayerNorm(h),nn.GELU())

    def molecular_features(self,atom_features,bond_features,edge_index,atom_drug):
        h=sum(emb(atom_features[:,j]) for j,emb in enumerate(self.atom_embeddings))/math.sqrt(len(self.atom_embeddings))
        if self.mode=='gnn':
            b=sum(emb(bond_features[:,j]) for j,emb in enumerate(self.bond_embeddings))/math.sqrt(len(self.bond_embeddings))
        for layer in self.layers:
            message=torch.zeros_like(h)
            if self.mode=='gnn':
                message.index_add_(0,edge_index[1],F.relu(h[edge_index[0]]+b))
            h=layer(h,message)
        pooled=h.new_zeros((self.ndrug,h.shape[1]))
        pooled.index_add_(0,atom_drug,h)
        pooled=pooled/self.atom_counts[:,None]
        return self.output(torch.cat([pooled,self.atom_counts.log1p()[:,None]],1))

    def forward(self,xd):
        if self.mode!='chem':
            return self.molecular_features(self.atom_features,self.bond_features,self.edge_index,self.atom_drug)
        h=self.input(xd)
        for layer in self.layers:h=layer(h,torch.zeros_like(h))
        return self.output(torch.cat([h,self.atom_counts.log1p()[:,None]],1))


class MolecularBranch(nn.Module):
    def __init__(self,hidden,xd_width,molecules,metadata,cfg):
        super().__init__();h=cfg['hidden']
        # These shared pieces are initialized first, identically for all three input modes.
        self.rna=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,h),nn.GELU())
        self.pair=nn.Sequential(nn.Linear(4*h,2*h),nn.GELU(),nn.Dropout(cfg['dropout']),nn.Linear(2*h,1))
        nn.init.zeros_(self.pair[-1].weight);nn.init.zeros_(self.pair[-1].bias)
        self.drug=DrugEncoder(xd_width,molecules,metadata,cfg)

    def forward(self,zm,xd,d):
        r=self.rna(zm);drug=self.drug(xd)[d]
        return self.pair(torch.cat([r,drug,r*drug,(r-drug).abs()],1)).view(-1)


class MolecularFusion(Model):
    def forward(self,eli,ps,qis,ease,dense):
        r,d=eli;xr,xd,Sr1,Sd1,A=dense
        zr,zd,zm,zdd,corr=self.encoder.forward_dense(xr,xd,Sr1,Sd1,A,r,d,ps.qvals(r,d))
        parts=[zm,zdd]
        parts += [lin(e) for lin,e in zip(self.ncn_lin,ps.qncn(r,d,{'miRNA':zr,'drug':zd},corr))]
        parts.append(self.pc(qis));parts.append(self.em(self.eb(ease)))
        h=F.leaky_relu(self.lin1(torch.cat(parts,-1)),.1)
        h=F.leaky_relu(self.lin2(h),.1)
        self.base_logits=self.lin3(h).view(-1)
        self.branch_logits=self.molecular(zm,xd,d)
        return self.base_logits+self.branch_logits


def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name=='control':return a
    # Preserve the exact reconstruction-only initialization before adding branch parameters.
    a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    if name=='unmasked10':return a
    molecules,metadata=load_molecules(device)
    a.model.__class__=MolecularFusion
    a.model.add_module('molecular',MolecularBranch(a.cfg.hidden,a.graph['drug'].x.shape[1],molecules,metadata,CONFIGS[name]).to(device))
    return a


def mask_generator(device,fold):
    return torch.Generator(device=device).manual_seed(20261006+fold)
