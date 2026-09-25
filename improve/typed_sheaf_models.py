"""Typed orthogonal sheaf message passing on the actual ncRNA/drug graph.

Inspired by DNSD (arXiv:2605.19021), using degree-normalized SO(2) transport,
per-stalk normalization, odd activations and residual gates. This is an
attributed heterogeneous link-prediction adaptation, not the paper's model.
Identity and scalar controls use the same node input and prediction head.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def rotate_stalk(x, angle, scalar=False):
    c, s = angle.cos().unsqueeze(-1), angle.sin().unsqueeze(-1)
    if scalar:
        return x * c.unsqueeze(-1)
    a, b = x[:, 0], x[:, 1]
    return torch.stack((c*a-s*b, s*a+c*b), dim=1)


def transport_aggregate(h, edges, angle, mode='orthogonal'):
    """Self-adjoint SO(2) adjacency normalized by physical graph degrees."""
    s, t = edges
    degree = torch.bincount(edges.flatten(), minlength=len(h)).to(h.dtype)
    scale = (degree[s].clamp_min(1)*degree[t].clamp_min(1)).rsqrt()
    if mode=='identity':
        toward_t,toward_s=h[s],h[t]
    else:
        toward_t=rotate_stalk(h[s],angle,scalar=mode=='scalar')
        toward_s=rotate_stalk(h[t],-angle,scalar=mode=='scalar')
    out=torch.zeros_like(h)
    out.index_add_(0,t,toward_t*scale[:,None,None])
    out.index_add_(0,s,toward_s*scale[:,None,None])
    return out


def physical_counts(edges, relation, n_nodes, n_relations):
    counts=torch.zeros(n_nodes*n_relations,device=edges.device)
    ones=torch.ones(edges.shape[1],device=edges.device)
    counts.index_add_(0,edges[0]*n_relations+relation,ones)
    counts.index_add_(0,edges[1]*n_relations+relation,ones)
    return torch.log1p(counts.reshape(n_nodes,n_relations))


class SheafLayer(nn.Module):
    def __init__(self, channels, relations, mode):
        super().__init__()
        assert channels%2==0
        self.mode=mode
        self.relation=nn.Embedding(relations,8)
        self.maps=nn.Sequential(nn.Linear(2*channels+8,32),nn.Tanh(),nn.Linear(32,2))
        nn.init.zeros_(self.maps[-1].weight);nn.init.zeros_(self.maps[-1].bias)
        self.feature=nn.Linear(channels//2,channels//2,bias=False)
        self.mix_stalk=nn.Parameter(torch.eye(2))
        self.gate=nn.Linear(channels,2)
        nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,-1.5)
        self.norm=nn.LayerNorm(channels//2)

    def forward(self,h,edges,relations):
        flat=h.flatten(1);s,t=edges
        if self.mode=='identity':
            angle=torch.zeros(len(s),device=h.device,dtype=h.dtype)
        else:
            maps=self.maps(torch.cat((flat[s],flat[t],self.relation(relations)),1))
            angle=np.pi*torch.tanh(maps[:,1]-maps[:,0])
        message=transport_aggregate(self.feature(h),edges,angle,self.mode)
        message=torch.einsum('ab,nbf->naf',self.mix_stalk,message)
        gate=torch.sigmoid(self.gate(flat))[:,:,None]
        return self.norm(h+gate*torch.tanh(message))


class TypedSheaf(nn.Module):
    def __init__(self, input_dims, node_counts, n_relations, channels=64,depth=4,mode='orthogonal'):
        super().__init__()
        self.n_relations=n_relations
        self.n_nodes=sum(node_counts)
        self.node_counts=node_counts
        self.project=nn.ModuleList([nn.Linear(d+n_relations,channels) for d in input_dims])
        self.layers=nn.ModuleList([SheafLayer(channels,n_relations,mode) for _ in range(depth)])
        self.input_norm=nn.LayerNorm(channels)
        self.dropout=nn.Dropout(.1)
        # Counts are the same structural input in every transport control.
        self.decoder=nn.Sequential(nn.Linear(4*channels+2*n_relations,128),nn.LeakyReLU(.1),
                                   nn.Dropout(.1),nn.Linear(128,64),nn.LeakyReLU(.1),nn.Linear(64,1))

    def encode(self,xs,edges,relations):
        counts=physical_counts(edges,relations,self.n_nodes,self.n_relations)
        projected=[];offset=0
        for x,proj,n in zip(xs,self.project,self.node_counts):
            # Normalize attributes without using outcome labels; counts stay on log scale.
            norm=x.square().mean(1,keepdim=True).clamp_min(1e-8).sqrt()
            projected.append(proj(torch.cat((x/norm,counts[offset:offset+n]),1)))
            offset+=n
        h=self.dropout(F.leaky_relu(self.input_norm(torch.cat(projected,0)),.1))
        h=h.reshape(self.n_nodes,2,-1)
        for layer in self.layers:h=layer(h,edges,relations)
        return h.flatten(1),counts

    def decode(self,h,counts,pairs):
        r,d=pairs
        z=torch.cat((h[r],h[d],h[r]*h[d],(h[r]-h[d]).abs(),counts[r],counts[d]),1)
        return self.decoder(z).squeeze(-1)

    def forward(self,xs,edges,relations,pairs):
        return self.decode(*self.encode(xs,edges,relations),pairs)


def configurations():
    return [dict(id=f'{mode}_d{depth}',mode=mode,depth=depth,channels=64,lr=.001,
                 weight_decay=.01,epochs=400,query_fraction=.5)
            for depth in (2,4) for mode in ('identity','scalar','orthogonal')]
