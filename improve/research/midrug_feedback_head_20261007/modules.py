"""Two exact query-deleted recurrent steps with block triangular feedback."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class GeometryFeedback(nn.Module):
    def __init__(self,nr,nd,mode,blocks=4,block_size=4):
        super().__init__();self.mode=mode;self.blocks=blocks;self.block_size=block_size
        self.hidden=blocks*block_size;self.off=block_size*(block_size-1)//2;self.coords=self.off+block_size
        self.rinput=nn.Sequential(nn.LayerNorm(nr),nn.Linear(nr,self.hidden))
        self.dinput=nn.Sequential(nn.LayerNorm(nd),nn.Linear(nd,self.hidden))
        self.phi=nn.Linear(self.hidden,self.hidden,bias=False)
        self.gate=nn.Linear(2*self.hidden,self.hidden)
        # Shared rank-m controller, with only fixed node-type signature.
        self.controller=nn.Sequential(nn.Linear(2*block_size+2,16),nn.SiLU(),nn.Linear(16,2*block_size*block_size+block_size))
        self.register_buffer('lower_indices',torch.tril_indices(block_size,block_size,-1),persistent=False)
        self.chunk_size=256

    def frame(self,z):
        m=self.block_size
        L=torch.diag_embed(z[...,self.off:].exp())
        L[...,self.lower_indices[0],self.lower_indices[1]]=z[...,:self.off]
        return L

    def coordinates(self,L):
        return torch.cat([L[...,self.lower_indices[0],self.lower_indices[1]],L.diagonal(dim1=-2,dim2=-1).log()],-1)

    def clip(self,z):
        return torch.cat([z[...,:self.off].clamp(-2,2),z[...,self.off:].clamp(-5,5)],-1)

    def step(self,x,z,nx,nz,valid,types,details=False):
        """Targets [T,d], neighbors [T,K,d]; duplicate edge events retain mass."""
        b,m=self.blocks,self.block_size
        L,NL=self.frame(z),self.frame(nz)
        h=(L@x.reshape(-1,b,m,1)).squeeze(-1)
        nh=(NL@nx.reshape(*nx.shape[:-1],b,m,1)).squeeze(-1)
        energy=(nh-h[:,None]).square().sum((-1,-2))
        weights=torch.softmax((-energy).masked_fill(~valid,-torch.inf),dim=1)
        px=self.phi(x);pn=self.phi(nx)
        env=NL@pn.reshape(*pn.shape[:-1],b,m,1)
        messages=torch.linalg.solve_triangular(L[:,None],env,upper=False).squeeze(-1)
        agg=(weights[:,:,None,None]*messages).sum(1).reshape(-1,self.hidden)
        rate=torch.sigmoid(self.gate(torch.cat([x,agg],-1)))
        newx=(1-rate)*x+rate*agg
        residual=messages-px.reshape(-1,1,b,m)
        if self.mode=='mean':
            mean=(weights[:,:,None,None]*residual).sum(1)
            C=mean[...,None]*mean[...,None,:]
        else:
            C=torch.einsum('tk,tkbi,tkbj->tbij',weights,residual,residual)
            if self.mode=='diagonal':C=torch.diag_embed(C.diagonal(dim1=-2,dim2=-1))
        eye=torch.eye(m,device=x.device,dtype=x.dtype)
        # Statistical ridge and fixed numerical jitter are recorded separately.
        C=C+1e-3*eye
        R=torch.linalg.cholesky(C+1e-5*eye)
        target=self.coordinates(R)
        cinput=torch.cat([x.reshape(-1,b,m),agg.reshape(-1,b,m),types[:,None,:].expand(-1,b,-1)],-1)
        controls=self.controller(cinput)
        U,V,d=controls.split([m*m,m*m,m],dim=-1)
        correction_matrix=U.reshape(-1,b,m,m)@V.reshape(-1,b,m,m).transpose(-1,-2)
        correction=torch.cat([correction_matrix[...,self.lower_indices[0],self.lower_indices[1]],d],-1)
        newz=self.clip(.5*z+.5*target+.1*correction)
        if details:return newx,newz,dict(weights=weights,messages=messages,residual=residual,moment=C,target=target,correction=correction)
        return newx,newz

    def prepare_graph(self,g):
        """Build full incoming lists; retain every original edge and existing self event."""
        nm=len(g['miRNA'].x);n=nm+len(g['drug'].x);offset={'miRNA':0,'drug':nm}
        incoming=[[] for _ in range(n)]
        for s,rel,d in g.edge_types:
            for u,v in g[s,rel,d].edge_index.T.cpu().tolist():incoming[v+offset[d]].append(u+offset[s])
        for i,ns in enumerate(incoming):
            if i not in ns:ns.append(i)
            ns.sort()
        k=max(map(len,incoming));device=g['miRNA'].x.device
        neighbors=torch.zeros(n,k,dtype=torch.long,device=device)
        valid=torch.zeros(n,k,dtype=torch.bool,device=device)
        for i,ns in enumerate(incoming):
            neighbors[i,:len(ns)]=torch.tensor(ns,device=device);valid[i,:len(ns)]=True
        self.register_buffer('neighbors',neighbors,persistent=False)
        self.register_buffer('valid',valid,persistent=False)
        associations=torch.zeros(nm,n-nm,dtype=torch.bool,device=device)
        e=g['miRNA','MiDrug','drug'].edge_index
        associations[e[0],e[1]]=True
        self.register_buffer('associations',associations,persistent=False)
        self.register_buffer('types',F.one_hot(torch.cat([torch.zeros(nm,dtype=torch.long,device=device),torch.ones(n-nm,dtype=torch.long,device=device)]),2).to(g['miRNA'].x.dtype),persistent=False)
        self.nm=nm

    def initial(self,xr,xd):
        x=torch.cat([self.rinput(xr),self.dinput(xd)])
        z=x.new_zeros(len(x),self.blocks,self.coords)
        return x,z

    def full_step(self,x,z,valid=None):
        valid=self.valid if valid is None else valid
        return self.step(x,z,x[self.neighbors],z[self.neighbors],valid,self.types)

    def query_chunk(self,pairs,x0,z0,x1,z1):
        # Interleave both endpoints; removing the target arc in each incoming list
        # changes only these nodes at the first step because x0/z0 are graph-free.
        endpoints=torch.stack([pairs[0],pairs[1]+self.nm],1)
        targets=endpoints.reshape(-1);other=endpoints.flip(1).reshape(-1)
        ns=self.neighbors[targets];valid=self.valid[targets]&(ns!=other[:,None])
        qx1,qz1=self.step(x0[targets],z0[targets],x0[ns],z0[ns],valid,self.types[targets])
        # At step two only target self events carry a changed source state. The
        # opposite endpoint has changed too, but all its target arcs are removed.
        own=ns==targets[:,None]
        nx=torch.where(own[:,:,None],qx1[:,None,:],x1[ns])
        nz=torch.where(own[:,:,None,None],qz1[:,None,:,:],z1[ns])
        qx2,qz2=self.step(qx1,qz1,nx,nz,valid,self.types[targets])
        h=(self.frame(qz2)@qx2.reshape(-1,self.blocks,self.block_size,1)).squeeze(-1)
        return h.reshape(pairs.shape[1],2*self.hidden)

    def forward(self,xr,xd,pairs):
        x0,z0=self.initial(xr,xd);x1,z1=self.full_step(x0,z0)
        observed=self.associations[pairs[0],pairs[1]]
        pos=torch.nonzero(observed).flatten();neg=torch.nonzero(~observed).flatten()
        out=x0.new_zeros(pairs.shape[1],2*self.hidden)
        # Every absent query uses exactly the same full graph, so share its
        # second step. The mask is graph membership, never the query label.
        if len(neg):
            x2,z2=self.full_step(x1,z1)
            h=(self.frame(z2)@x2.reshape(-1,self.blocks,self.block_size,1)).squeeze(-1).reshape(-1,self.hidden)
            out=out.index_copy(0,neg,torch.cat([h[pairs[0,neg]],h[pairs[1,neg]+self.nm]],-1))
        values=[]
        for part in pairs[:,pos].split(self.chunk_size,dim=1):
            if not part.shape[1]:continue
            if self.training and torch.is_grad_enabled():
                values.append(checkpoint(self.query_chunk,part,x0,z0,x1,z1,use_reentrant=True))
            else:values.append(self.query_chunk(part,x0,z0,x1,z1))
        if len(pos):out=out.index_copy(0,pos,torch.cat(values))
        return out

    def explicit(self,xr,xd,pair):
        """Whole-graph physical masking reference, used only for checks."""
        targets=[int(pair[0]),int(pair[1])+self.nm]
        valid=self.valid.clone()
        for t,o in [targets,targets[::-1]]:valid[t]&=self.neighbors[t]!=o
        x,z=self.initial(xr,xd)
        for _ in range(2):x,z=self.full_step(x,z,valid)
        h=(self.frame(z[targets])@x[targets].reshape(2,self.blocks,self.block_size,1)).squeeze(-1)
        return h.reshape(1,2*self.hidden)
