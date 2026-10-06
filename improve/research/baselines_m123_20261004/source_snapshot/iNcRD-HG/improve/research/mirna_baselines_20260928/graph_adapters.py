"""Graph adapters. Local ports are explicitly identified in result metadata."""
import ast, types
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from bench_data import BASE

def classes_from(path,names,globals_dict):
    tree=ast.parse(path.read_text())
    tree.body=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
    mod=types.ModuleType(path.stem); mod.__dict__.update(globals_dict)
    exec(compile(tree,str(path),'exec'),mod.__dict__)
    return mod

def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)

def gip(x):
    sq=(x*x).sum(1); dist=np.maximum(sq[:,None]+sq[None,:]-2*x@x.T,0)
    return np.exp(-dist/max(sq.mean(),1e-8)).astype(np.float32)

def block(x,y):
    z=np.zeros((len(x)+len(y),x.shape[1]+y.shape[1]),np.float32)
    z[:len(x),:x.shape[1]]=unit(x); z[len(x):,x.shape[1]:]=unit(y)
    return z

def normalized_sparse(a,dev):
    a=sp.csr_matrix(a)+sp.eye(a.shape[0]); d=np.asarray(a.sum(1)).ravel()**-0.5
    a=(sp.diags(d)@a@sp.diags(d)).tocoo()
    return torch.sparse_coo_tensor(torch.tensor(np.array([a.row,a.col]),dtype=torch.long),
                                  torch.tensor(a.data,dtype=torch.float32),a.shape).coalesce().to(dev)

class GraphAdapter:
    def __init__(self,data,args):
        self.data=data; self.args=args; self.dev=args.device; self.kind=args.model
        xr,xd=data['xr'],data['xd']; self.nr=len(xr); self.nd=len(xd)
        p=data['train_pairs'][data['train_y']==1]
        R=np.zeros((self.nr,self.nd),np.float32); R[p[:,0],p[:,1]]=1
        self.pos=torch.tensor(p.T,dtype=torch.long,device=self.dev); self.pos[1]+=self.nr
        self.edges=torch.cat([self.pos,self.pos.flip(0)],1)
        A=sp.bmat([[None,sp.csr_matrix(R)],[sp.csr_matrix(R.T),None]],format='csr')
        self.adj=normalized_sparse(A,self.dev)
        self.x=torch.tensor(block(xr,xd),device=self.dev)
        self.pairs={s:torch.tensor(data[s+'_pairs'].T,dtype=torch.long,device=self.dev) for s in ('train','val','test')}
        self.y=torch.tensor(data['train_y'],dtype=torch.float,device=self.dev)
        local=BASE/'baseline_repro_20260918'
        baseglobals=dict(torch=torch,nn=nn,F=F,GCNConv=GCNConv,DROPOUT=0.1,HID=128)
        if self.kind=='gcmdr':
            m=classes_from(local/'repro_gcmdr_ncdr.py',{'GCMDR'},baseglobals)
            self.model=m.GCMDR(self.nr+self.nd,self.x.shape[1],25).to(self.dev)
            self.note='GCMDR local PyTorch port: graph convolution + node bias + bilinear decoder; kmer/Morgan content replaces original auxiliary features.'
        elif self.kind=='ammgc':
            m=classes_from(local/'repro_ammgc_ncdr.py',{'GCN','AMMGC'},baseglobals)
            sr,sd=gip(R),gip(R.T)
            self.views=[torch.tensor(block(a,b),device=self.dev) for a,b in ((xr,xd),(xr,sd),(sr,xd),(sr,sd))]
            self.model=m.AMMGC([v.shape[1] for v in self.views],256).to(self.dev)
            self.note='AMMGC local PyTorch port: four GCN views and attention fusion; train-only GIP + kmer/Morgan views.'
        elif self.kind=='lagcn':
            m=classes_from(BASE/'msffmdi_baselines_20260920/msffmdi_models.py',{'LAGCN'},baseglobals)
            self.model=m.LAGCN(xr.shape[1],xd.shape[1],hidden=64,layers=2).to(self.dev); self.model.nr=self.nr
            self.xr=torch.tensor(unit(xr),device=self.dev); self.xd=torch.tensor(unit(xd),device=self.dev)
            sims=[]
            for x in (xr,xd):
                sim=unit(x)@unit(x).T; np.fill_diagonal(sim,0)
                a=np.zeros_like(sim); ix=np.argsort(-sim,axis=1)[:,:10]; np.put_along_axis(a,ix,1,axis=1)
                sims.append(np.maximum(a,a.T))
            aa=sp.bmat([[sp.csr_matrix(sims[0]),sp.csr_matrix(R)],[sp.csr_matrix(R.T),sp.csr_matrix(sims[1])]],format='coo')
            self.edges=torch.tensor(np.stack([aa.row,aa.col]),device=self.dev).long()
            self.ew=torch.tensor(aa.data,device=self.dev).float()
            self.note='LAGCN architecture reimplementation (local): layer attention over 2 GCN layers; 10-NN sequence/structure similarity and training associations; not an official-code reproduction.'
        elif self.kind=='jmssmma':
            from torch_sparse import SparseTensor
            from torch_geometric.nn import Linear,GINConv
            m=classes_from(local/'repos/JMSSMMA/model.py',{'GNNEncoder','EdgeDecoder','DegreeDecoder'},
                           dict(torch=torch,nn=nn,F=F,Linear=Linear,GINConv=GINConv,SparseTensor=SparseTensor))
            self.x=torch.tensor(block(gip(R),gip(R.T)),device=self.dev)
            self.model=nn.ModuleDict(dict(encoder=m.GNNEncoder(self.x.shape[1],64,128),
                                         edge=m.EdgeDecoder(128,64,1),degree=m.DegreeDecoder(128,64,1))).to(self.dev)
            self.note='JMSSMMA author GIN encoder/edge+degree decoders; mask 40% training positives; fixed bipartite training negatives replace author homogeneous random negatives; GIP built only from training positives.'
        else: raise ValueError(self.kind)

    def embeddings(self):
        if self.kind=='gcmdr': return self.model(self.adj,self.x)
        if self.kind=='ammgc': return self.model(self.adj,self.views)
        if self.kind=='lagcn': return torch.cat(self.model(self.xr,self.xd,self.edges,self.ew),0)
        return self.model['encoder'](self.x,self.edges)

    def score(self,z,p):
        if self.kind=='jmssmma':
            e=p.clone(); e[1]+=self.nr
            return self.model['edge'](z,e).flatten()
        if self.kind=='gcmdr': return ((z[p[0]]@self.model.W3)*z[p[1]+self.nr]).sum(-1)
        return (z[p[0]]*z[p[1]+self.nr]).sum(-1)

    def train_epoch(self,opt,epoch):
        opt.zero_grad()
        if self.kind=='jmssmma':
            mask=torch.rand(self.pos.shape[1],device=self.dev)<0.4
            if not mask.any(): mask[0]=True
            rem=self.pos[:,~mask]; rem=torch.cat([rem,rem.flip(0)],1)
            z=self.model['encoder'](self.x,rem)
            masked=self.pos[:,mask]
            neg=self.pairs['train'][:,self.y==0]
            idx=torch.randint(neg.shape[1],(masked.shape[1],),device=self.dev)
            neg=neg[:,idx].clone(); neg[1]+=self.nr
            pp=self.model['edge'](z,masked).flatten(); pn=self.model['edge'](z,neg).flatten()
            loss=F.binary_cross_entropy_with_logits(pp,torch.ones_like(pp))+F.binary_cross_entropy_with_logits(pn,torch.zeros_like(pn))
            deg=torch.bincount(masked[1],minlength=self.nr+self.nd).float()
            loss=loss+0.007*F.mse_loss(self.model['degree'](z).flatten(),deg)
        else:
            z=self.embeddings(); loss=F.binary_cross_entropy_with_logits(self.score(z,self.pairs['train']),self.y)
        loss.backward(); nn.utils.clip_grad_norm_(self.model.parameters(),1.0); opt.step()
        return float(loss.detach())

    def predict(self,part):
        return torch.sigmoid(self.score(self.embeddings(),self.pairs[part])).cpu().numpy()
