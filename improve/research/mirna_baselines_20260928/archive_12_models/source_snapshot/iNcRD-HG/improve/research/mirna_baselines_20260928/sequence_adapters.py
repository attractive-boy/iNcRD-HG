"""Content and multi-view baselines using the existing, attributed model definitions."""
import sys
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from bench_data import BASE

DLST=BASE/'dlstmda_baselines_20260919'
sys.path.insert(0,str(DLST))
from common import seq2int, smiles2int, smile_to_graph
from models import GraphDTA_GCN, SubMDTA, MLDTI
sys.path.insert(0,str(BASE/'plmf_strong_baselines_20260922'))
from models_extra import GCNNetmuti,DLMVF

def vocabulary(smiles,seqs):
    """Author WL fingerprint scheme, with isolated atoms retained in disconnected drugs."""
    from rdkit import Chem
    ad=defaultdict(lambda:len(ad)); bd=defaultdict(lambda:len(bd))
    fd=defaultdict(lambda:len(fd)); ed=defaultdict(lambda:len(ed)); wd=defaultdict(lambda:len(wd))
    compounds=[]
    for s in smiles:
        mol=Chem.AddHs(Chem.MolFromSmiles(s))
        nodes=[ad[(a.GetSymbol(),'aromatic') if a.GetIsAromatic() else a.GetSymbol()] for a in mol.GetAtoms()]
        edges={i:[] for i in range(len(nodes))}
        for b in mol.GetBonds():
            i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); k=bd[str(b.GetBondType())]
            edges[i].append((j,k)); edges[j].append((i,k))
        for _ in range(2):
            new=[fd[(nodes[i],tuple(sorted((nodes[j],e) for j,e in edges[i])))] for i in range(len(nodes))]
            edges={i:[(j,ed[(tuple(sorted((new[i],new[j]))),e)]) for j,e in es] for i,es in edges.items()}
            nodes=new
        compounds.append((torch.tensor(nodes),torch.tensor(Chem.GetAdjacencyMatrix(mol),dtype=torch.float)))
    words=[]
    for s in seqs:
        s='-'+s+'='; words.append([wd[s[i:i+3]] for i in range(len(s)-2)])
    return compounds,words,len(fd),len(wd)

class SequenceAdapter:
    def __init__(self,data,args):
        self.data=data; self.args=args; self.dev=args.device; self.kind=args.model
        self.seq=torch.tensor(np.stack([seq2int(s) for s in data['sequences']]),dtype=torch.long,device=self.dev)
        self.smi=torch.tensor(np.stack([smiles2int(s) for s in data['smiles']]),dtype=torch.long,device=self.dev)
        self.pairs={s:torch.tensor(data[s+'_pairs'],device=self.dev) for s in ('train','val','test')}
        self.y=torch.tensor(data['train_y'],dtype=torch.float,device=self.dev)
        self.loaders={}
        if self.kind=='mldti':
            self.model=MLDTI().to(self.dev)
            self.note='ML-DTI author mutual-attention CNN adapted from protein to miRNA and binary BCE; mature sequence truncated/padded to 24 nt.'
            return
        if self.kind=='gcnnmma':
            sys.path.insert(0,str(BASE/'plmf_baselines_20260920'))
            from gcnnma_plmf import GCNNMMA
            self.comp,self.words,nf,nw=vocabulary(data['smiles'],data['sequences'])
            self.lmax=max(map(len,self.words)); self.model=GCNNMMA(nf,nw).to(self.dev)
            self.note='GCNNMMA batched author-architecture port, WL radius=2, ngram=3; fix isolated-atom fingerprint indexing; unlabeled vocabulary over known entities.'
            return
        graphs=[]
        for s in data['smiles']:
            _,x,e=smile_to_graph(s)
            ei=torch.tensor(np.asarray(e),dtype=torch.long).reshape(-1,2).t().contiguous()
            graphs.append((torch.tensor(np.asarray(x),dtype=torch.float),ei))
        for part in ('train','val','test'):
            samples=[]
            for (i,j),y in zip(data[part+'_pairs'],data[part+'_y']):
                x,e=graphs[j]; d=Data(x=x,edge_index=e,y=torch.tensor([y],dtype=torch.float))
                d.target=self.seq[i].cpu().unsqueeze(0); d.seqdrug=self.smi[j].cpu().unsqueeze(0)
                if self.kind=='dlmvf':
                    d.gene=torch.tensor(data['xr'][i],dtype=torch.float).unsqueeze(0)
                    d.row_indices=torch.tensor([i]); d.col_indices=torch.tensor([j])
                samples.append(d)
            # Merge a singleton training tail into the preceding batch (BatchNorm).
            batch=args.batch
            self.loaders[part]=DataLoader(samples,batch_size=batch,shuffle=part=='train',drop_last=part=='train' and len(samples)%batch==1)
        if self.kind=='graphdta': self.model=GraphDTA_GCN()
        elif self.kind=='submdta': self.model=SubMDTA()
        elif self.kind=='dlstmda': self.model=GCNNetmuti()
        elif self.kind=='dlmvf':
            nr,nd=len(data['xr']),len(data['xd'])
            self.model=DLMVF(drug_offset=nr,gene_dim=data['xr'].shape[1])
            p=data['train_pairs'][data['train_y']==1]; e=torch.tensor(p.T.copy()).long(); e[1]+=nr
            rng=np.random.RandomState(args.seed); x=rng.normal(size=(nr+nd,256)); x/=x.sum(1,keepdims=True)
            self.graph=Data(x=torch.tensor(x,dtype=torch.float),edge_index=torch.cat([e,e.flip(0)],1)).to(self.dev)
        else: raise ValueError(self.kind)
        self.model=self.model.to(self.dev)
        self.note={
            'graphdta':'GraphDTA GCN adapted to miRNA length 24 and binary association BCE.',
            'submdta':'SubMDTA prediction architecture adapted to miRNA; substructure self-supervised pretraining omitted (adapted, not full official model).',
            'dlstmda':'DLST-MDA author GCNNetmuti network; shared 78-d atom/SMILES/24-nt mature miRNA preprocessing.',
            'dlmvf':'DLMVF author network; miRNA gene-network view replaced by content kmer (84 dims); association graph uses only training positives; random graph features use author row normalization.',
        }[self.kind]

    def tensor_batches(self,part,shuffle=False):
        n=len(self.data[part+'_y']); order=np.random.permutation(n) if shuffle else np.arange(n)
        for start in range(0,n,self.args.batch):
            ix=order[start:start+self.args.batch]; pairs=self.data[part+'_pairs'][ix]
            if self.kind=='mldti':
                ij=self.pairs[part][ix]
                yield ix,(self.seq[ij[:,0]],self.smi[ij[:,1]])
            else:
                size=max(len(self.comp[j][0]) for i,j in pairs); b=len(ix)
                fp=torch.zeros(b,size,dtype=torch.long); adj=torch.zeros(b,size,size); mask=torch.zeros(b,size)
                words=torch.zeros(b,self.lmax,dtype=torch.long)
                for row,(i,j) in enumerate(pairs):
                    f,a=self.comp[j]; count=len(f); fp[row,:count]=f; adj[row,:count,:count]=a; mask[row,:count]=1
                    words[row,:len(self.words[i])]=torch.tensor(self.words[i])
                yield ix,tuple(v.to(self.dev) for v in (fp,adj,mask,words))

    def train_epoch(self,opt,epoch):
        total=0.; n=0
        if self.kind=='gcnnmma' and epoch%10==0:
            for group in opt.param_groups: group['lr']*=0.5
        if self.kind in ('mldti','gcnnmma'):
            for ix,inputs in self.tensor_batches('train',True):
                opt.zero_grad(); z=self.model(*inputs)
                loss=F.cross_entropy(z,self.y[ix].long()) if self.kind=='gcnnmma' else F.binary_cross_entropy_with_logits(z.flatten(),self.y[ix])
                loss.backward(); opt.step(); total+=float(loss.detach())*len(ix); n+=len(ix)
        else:
            for batch in self.loaders['train']:
                batch=batch.to(self.dev); opt.zero_grad()
                z=self.model(batch,self.graph) if self.kind=='dlmvf' else self.model(batch)
                loss=(F.binary_cross_entropy if self.kind in ('dlstmda','dlmvf') else F.binary_cross_entropy_with_logits)(z.flatten(),batch.y.flatten())
                loss.backward(); opt.step(); total+=float(loss.detach())*len(batch.y); n+=len(batch.y)
        return total/n

    def predict(self,part):
        out=[]
        if self.kind in ('mldti','gcnnmma'):
            for ix,inputs in self.tensor_batches(part):
                z=self.model(*inputs)
                p=z.softmax(-1)[:,1] if self.kind=='gcnnmma' else z.flatten().sigmoid()
                out.append(p.cpu().numpy())
        else:
            for b in self.loaders[part]:
                b=b.to(self.dev); z=self.model(b,self.graph) if self.kind=='dlmvf' else self.model(b)
                p=z.flatten() if self.kind in ('dlstmda','dlmvf') else z.flatten().sigmoid()
                out.append(p.cpu().numpy())
        return np.concatenate(out)
