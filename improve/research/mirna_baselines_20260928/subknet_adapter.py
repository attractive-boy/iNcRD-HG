"""Author SubKNet model/extraction, fixed validation selection and ordered full scoring."""
import importlib.util, sys, types
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from bench_data import BASE

CODE=BASE/'plmf_strong_baselines_20260922/repos/SubKNet/code'

def onehot_linear(self,x):
    """Exactly the same linear map on one-hot or zero rows, without dense one-hot allocation."""
    if x.dtype==torch.long:
        table=torch.cat([self.weight.t(),self.weight.new_zeros(1,self.out_features)],0)
        return nn.functional.embedding(x,table)+self.bias
    return nn.functional.linear(x,self.weight,self.bias)

def compact_rw_forward(self,adj,features):
    # Training retains the author's elementwise stochastic dropout computation.
    if self.training:return self._author_forward(adj,features)
    x=nn.functional.relu(self.fc_in(features))
    z=self.features_hidden.clamp(0,1); x0=x; z0=z
    ah=z.new_zeros(self.size_graph_filter,self.size_graph_filter,self.out_dim)
    idx=torch.triu_indices(self.size_graph_filter,self.size_graph_filter,1,device=z.device)
    ah[idx[0],idx[1],:]=self.adj_hidden.clamp(-1,1).relu();ah=ah+ah.transpose(0,1)
    outputs=[]
    for step in range(self.max_step):
        if step:
            x=torch.bmm(adj,x);z=torch.einsum('abd,bcd->acd',ah,z)
        # sum_{m,n} (z0_m . x0_n)(zt_m . xt_n) via their small feature Gram matrices.
        xx=torch.einsum('bnc,bnd->bcd',x0,x)
        zz=torch.einsum('mco,mdo->cdo',z0,z)
        outputs.append(torch.einsum('bcd,cdo->bo',xx,zz)/(self.size_graph_filter*features.shape[1]))
    diversity=.5*self._diversity_term(z0.permute(2,0,1).mean(1))+.5*self._diversity_term(ah.permute(2,0,1).flatten(1))
    return sum(outputs)/len(outputs),diversity

def module(name,file):
    spec=importlib.util.spec_from_file_location(name,str(file)); m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m

class SubKNetAdapter:
    def __init__(self,data,args):
        self.data=data; self.args=args; self.dev=args.device
        cfg=SimpleNamespace(types='hete',gcn_layer=1,lambda1=0.5,lambda2=0.5,lambda3=0.5)
        self.cfg=cfg
        sys.path.insert(0,str(CODE))
        extract=module('subknet_extract',CODE/'subgraph_extraction.py')
        extract.args_parser=lambda:cfg
        net=module('subknet_model',CODE/'model.py')
        nr,nd=len(data['xr']),len(data['xd']); n=nr+nd
        r=np.zeros((nr,nd),np.float32); p=data['train_pairs'][data['train_y']==1]; r[p[:,0],p[:,1]]=1
        self.global_adj=torch.tensor(np.block([[np.zeros((nr,nr)),r],[r.T,np.zeros((nd,nd))]]),device=self.dev).float()
        ident=np.eye(n,dtype=np.float32); fm,fd=ident[:nr],ident[nr:]
        self.features=[]
        for x in (data['xr'],data['xd']):
            x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-9)
            self.features.append(PCA(n_components=min(128,nr,nd),svd_solver='full').fit_transform(x@x.T))
        # Author extraction uses numpy random permutations. Seed again after imports.
        from engine import seed_all
        seed_all(args.seed)
        self.graphs={}
        for part in ('train','val','test'):
            self.graphs[part]=[extract.subgraph_extraction_labeling((int(i),int(j)),r,h=1,featM=fm,featD=fd,g_label=int(y))
                               for (i,j),y in zip(data[part+'_pairs'],data[part+'_y'])]
        size=max(2,int(np.mean([g[0].shape[0] for g in self.graphs['train']])))
        self.model=net.Model(self.features[0].shape[1],self.features[1].shape[1],n,2,cfg,
                            max_step=2,dropout_rate=0.2,size_graph_filter=[size]).to(self.dev)
        self.compact=getattr(args,'compact_subknet',False)
        if self.compact:
            layer=self.model.ker_layers[0].fc_in
            layer.forward=types.MethodType(onehot_linear,layer)
            rw=self.model.ker_layers[0]
            rw._author_forward=rw.forward
            rw.forward=types.MethodType(compact_rw_forward,rw)
        self.note='SubKNet author enclosing-subgraph extraction, RW kernel, global DenseGCN and composite loss; content cosine PCA instead of original similarity tables; training-only global graph; preserves author softmax-plus-cross-entropy objective; all validation/test rows scored in original order.'

    def batches(self,part,shuffle=False):
        n=len(self.graphs[part]); order=np.random.permutation(n) if shuffle else np.arange(n)
        chunks=[order[i:i+self.args.batch] for i in range(0,n,self.args.batch)]
        if len(chunks)>1 and len(chunks[-1])==1:
            chunks[-2]=np.r_[chunks[-2],chunks[-1]];chunks.pop()
        # Author padding is constant across the whole split. Keep this for numerical parity.
        maxn=max(g[0].shape[0] for g in self.graphs[part]); dim=self.graphs[part][0][1].shape[1]
        for ix in chunks:
            adj=torch.zeros(len(ix),maxn,maxn)
            feat=torch.full((len(ix),maxn),dim,dtype=torch.long) if self.compact else torch.zeros(len(ix),maxn,dim)
            for row,i in enumerate(ix):
                g=self.graphs[part][i]; size=g[0].shape[0]
                adj[row,:size,:size]=torch.tensor(g[0].toarray())
                if self.compact:
                    indices=g[1].tocsr()
                    assert np.all(np.diff(indices.indptr)==1) and np.all(indices.data==1)
                    feat[row,:size]=torch.tensor(indices.indices,dtype=torch.long)
                else:feat[row,:size]=torch.tensor(g[1].toarray())
            yield ix,adj.to(self.dev),feat.to(self.dev),self.data[part+'_pairs'][ix]

    def forward(self,adj,feat,pairs):
        return self.model(self.global_adj,adj,feat,self.features[0],self.features[1],pairs,self.cfg)

    def train_epoch(self,opt,epoch):
        total=0.; n=0
        for ix,adj,feat,pairs in self.batches('train',True):
            opt.zero_grad(); out,div,local,glob=self.forward(adj,feat,pairs)
            y=torch.tensor(self.data['train_y'][ix],device=self.dev).long()
            loss=nn.functional.cross_entropy(out,y)+0.5*div+0.5*nn.functional.cross_entropy(local,y)+0.5*nn.functional.cross_entropy(glob,y)
            loss.backward(); opt.step(); total+=float(loss.detach())*len(ix);n+=len(ix)
        return total/n

    def predict(self,part):
        pred=np.empty(len(self.data[part+'_y']),np.float64)
        for ix,adj,feat,pairs in self.batches(part):pred[ix]=self.forward(adj,feat,pairs)[0][:,1].cpu().numpy()
        return pred
