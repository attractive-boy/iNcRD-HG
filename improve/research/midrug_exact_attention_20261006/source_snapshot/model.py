"""Positive normalized gates plus exact first/second-layer query-edge corrections."""
import torch
from torch import nn
from torch.nn import functional as F
from common import Adapter,Reconstruction,make_graph
from events import EventIndex
from incrd_hg_m123.encoder import QueryHollowEncoder
from incrd_hg_m123.pair_channels import PairStruct

CONFIGS={
    'target':dict(scope='target',alpha=0.,reconstruction_weight=0.),
    'all':dict(scope='all',alpha=0.,reconstruction_weight=0.),
    'target_rec10':dict(scope='target',alpha=0.,reconstruction_weight=10.),
    'all_rec10':dict(scope='all',alpha=0.,reconstruction_weight=10.),
    'all05':dict(scope='all',alpha=.5,reconstruction_weight=0.),
    'all05_rec10':dict(scope='all',alpha=.5,reconstruction_weight=10.),
}


class Gate(nn.Module):
    def __init__(self,source_dim,target_dim,width=16,enabled=True):
        super().__init__();self.enabled=enabled
        self.source=nn.Linear(source_dim,width,bias=False);self.target=nn.Linear(target_dim,width,bias=False)
        self.score=nn.Linear(width,1,bias=False);nn.init.zeros_(self.score.weight)
        if not enabled:self.requires_grad_(False)
    def project(self,x,side):
        return getattr(self,side)(F.layer_norm(x,(x.shape[-1],)))
    def score_projected(self,source,target):
        if not self.enabled:return source.new_full((len(source),),.5)
        # Strict positive bounds keep subtraction near a singleton receiver well conditioned.
        return .1+.8*torch.sigmoid(self.score(F.leaky_relu(source+target,.2)).view(-1))


class AttentionEncoder(QueryHollowEncoder):
    def configure(self,scope,alpha):
        self.scope,self.alpha=scope,alpha
        self.gates1=nn.ModuleDict();self.gates2=nn.ModuleDict()
        for e in self.graph.edge_types:
            name='__'.join(e);enabled=scope=='all' or (scope=='target' and e[0]!=e[2])
            for number,convs in [(1,self.original.conv1),(2,self.original.conv2)]:
                c=convs[name]
                getattr(self,'gates'+str(number))[name]=Gate(c.lin_msg.weight.shape[1],c.lin_self.weight.shape[1],enabled=enabled).to(c.lin_msg.weight.device)
        self.events=EventIndex(self.graph)

    def aggregate(self,degree,mass,numerator):
        # The degree mask gives exactly zero for genuinely empty neighborhoods.
        scale=torch.where(degree>0,degree.clamp(min=1).pow(1-self.alpha)/mass.clamp(min=1e-8),torch.zeros_like(mass))
        return scale[:,None]*numerator

    def full_layer(self,convs,x,number,edges=None):
        edges=self.graph.edge_index_dict if edges is None else edges
        out={};cache={}
        for e in self.graph.edge_types:
            name='__'.join(e);c=convs[name];gate=getattr(self,'gates'+str(number))[name]
            s,d=edges[e];n=len(x[e[2]])
            source=gate.project(x[e[0]],'source');target=gate.project(x[e[2]],'target')
            g=gate.score_projected(source[s],target[d]);value=c.lin_msg(x[e[0]])
            degree=torch.bincount(d,minlength=n).to(value.dtype)
            mass=value.new_zeros(n).index_add(0,d,g)
            numerator=value.new_zeros((n,value.shape[1])).index_add(0,d,g[:,None]*value[s])
            message=self.aggregate(degree,mass,numerator)
            out[e[2]]=out.get(e[2],0)+message+c.lin_self(x[e[2]])
            cache[e]=dict(degree=degree,mass=mass,numerator=numerator,message=message,
                          gate=g,value=value,source_proj=source,target_proj=target)
        return {n:v/self.counts[n] for n,v in out.items()},cache

    def layer(self,convs,x):
        number=1 if convs is self.original.conv1 else 2
        return self.full_layer(convs,x,number)[0]

    def forward(self,x,edges):
        p1,_=self.full_layer(self.original.conv1,x,1,edges)
        h1={n:F.leaky_relu(v,.1) for n,v in p1.items()}
        return self.full_layer(self.original.conv2,h1,2,edges)[0]

    def forward_dense(self,xr,xd,Sr1,Sd1,A,r=None,d=None,q=None):
        c1,c2=self.original.conv1,self.original.conv2
        p1,cache1=self.full_layer(c1,self.graph.x_dict,1)
        h1={n:F.leaky_relu(v,.1) for n,v in p1.items()}
        z,cache2=self.full_layer(c2,h1,2)
        if r is None:return z['miRNA'],z['drug']
        if not bool((q>0).any()):return z['miRNA'],z['drug'],z['miRNA'][r],z['drug'][d],dict(updates=[])
        endpoint={'miRNA':r,'drug':d};dh={}
        for e in [('miRNA','MiDrug','drug'),('drug','rev_MiDrug','miRNA')]:
            src,dst=endpoint[e[0]],endpoint[e[2]];cache=cache1[e]
            g=self.gates1['__'.join(e)].score_projected(cache['source_proj'][src],cache['target_proj'][dst])
            new_mass=cache['mass'][dst]-q*g
            new_num=cache['numerator'][dst]-(q*g)[:,None]*cache['value'][src]
            changed=self.aggregate(cache['degree'][dst]-q,new_mass,new_num)
            shift=(changed-cache['message'][dst])/self.counts[e[2]]
            dh[e[2]]=F.leaky_relu(p1[e[2]][dst]+shift,.1)-h1[e[2]][dst]
        geometry=self.events.get(r,d,q);updates=[]
        endpoint_delta={n:torch.zeros_like(h1[n][idx]) for n,idx in endpoint.items()}
        for e in self.graph.edge_types:
            name='__'.join(e);c=c2[name];gate=self.gates2[name];cache=cache2[e];ev=geometry[e]
            eq=ev['query'];gq,gn=ev['group_query'],ev['group_target'];ng=len(gq)
            if ng:
                src_endpoint,dst_endpoint=endpoint[e[0]],endpoint[e[2]]
                source_shift=gate.project(h1[e[0]][src_endpoint]+dh[e[0]],'source')-cache['source_proj'][src_endpoint]
                target_shift=gate.project(h1[e[2]][dst_endpoint]+dh[e[2]],'target')-cache['target_proj'][dst_endpoint]
                src=cache['source_proj'][ev['source']]+ev['source_is_query'][:,None]*source_shift[eq]
                dst=cache['target_proj'][ev['target']]+ev['target_is_query'][:,None]*target_shift[eq]
                new_gate=gate.score_projected(src,dst)*~ev['removed']
                old_gate=cache['gate'][ev['edge']];difference=new_gate-old_gate
                dm=cache['mass'].new_zeros(ng).index_add(0,ev['group'],difference)
                dn=cache['numerator'].new_zeros((ng,cache['numerator'].shape[1]))
                dn=dn.index_add(0,ev['group'],difference[:,None]*cache['value'][ev['source']])
                value_shift=F.linear(dh[e[0]],c.lin_msg.weight)
                dn=dn.index_add(0,ev['group'],(new_gate*ev['source_is_query'])[:,None]*value_shift[eq])
                removed_degree=q[gq]*(gn==endpoint[e[2]][gq]) if e[0]!=e[2] else torch.zeros_like(q[gq])
                message=self.aggregate(cache['degree'][gn]-removed_degree,cache['mass'][gn]+dm,cache['numerator'][gn]+dn)
                delta=(message-cache['message'][gn])/self.counts[e[2]]
                updates.append((e[2],gq,gn,delta))
                mask=gn==endpoint[e[2]][gq]
                endpoint_delta[e[2]]=endpoint_delta[e[2]].index_add(0,gq,delta*mask[:,None])
            # Explicit root projections change only at the query endpoints.
            endpoint_delta[e[2]]=endpoint_delta[e[2]]+F.linear(dh[e[2]],c.lin_self.weight)/self.counts[e[2]]
        return z['miRNA'],z['drug'],z['miRNA'][r]+endpoint_delta['miRNA'],z['drug'][d]+endpoint_delta['drug'],dict(updates=updates)


class AttentionPairStruct(PairStruct):
    def qncn(self,r,d,z,corr=None):
        if corr is None:return super().qncn(r,d,z,None)
        q=self.qvals(r,d)[:,None]
        def pool(w,node):
            den=w.sum(1,keepdim=True)+1
            value=w@z[node]/den
            for kind,query,index,delta in corr['updates']:
                if kind==node:
                    correction=(w[query,index]/den[query,0])[:,None]*delta
                    value=value.index_add(0,query,correction)
            return value
        Ad=self.A[:,d].t();Ar=self.A[r]
        wr=Ad*(self.Cm[r]-q*Ad);wd=Ar*(self.Cd[:,d].t()-q*Ar)
        ix=torch.arange(len(r),device=r.device);wr[ix,r]=0;wd[ix,d]=0
        return [pool(wr,'miRNA'),pool(wd,'drug'),pool(Ad*self.Sr[r],'miRNA'),pool(Ar*self.Sd[:,d].t(),'drug')]


def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name=='control':return a
    cfg=dict(scope='uniform',alpha=0.,reconstruction_weight=0.) if name=='identity' else (
        dict(scope='uniform',alpha=0.,reconstruction_weight=10.) if name=='unmasked10' else CONFIGS[name])
    if cfg['reconstruction_weight']:a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    if name=='unmasked10':return a
    a.model.encoder.__class__=AttentionEncoder;a.model.encoder.configure(cfg['scope'],cfg['alpha'])
    a.ps.__class__=AttentionPairStruct
    return a


def mask_generator(device,fold):return torch.Generator(device=device).manual_seed(20261006+fold)
