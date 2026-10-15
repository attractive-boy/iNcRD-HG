"""Masked neighbor-context prediction of centered latent targets, auxiliary only."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import HERE,Adapter,make_graph,Reconstruction

CONFIGS={'block_ema':dict(mask='block',momentum=.99),
         'random_ema':dict(mask='random',momentum=.99),
         'block_current':dict(mask='block',momentum=0.)}


def encode(encoder,convs,inputs):
    h=encoder.layer(convs.conv1,inputs)
    return encoder.layer(convs.conv2,{n:F.leaky_relu(v,.1) for n,v in h.items()})


def normalized_target(h):
    # Center within the node type: otherwise the original summed-message mean
    # makes predicting a common direction an almost-trivial latent objective.
    return F.normalize(h-h.mean(0,keepdim=True),dim=-1,eps=1e-8)


def cosine_loss(pred,target):
    valid=target.square().sum(-1)>1e-12
    if not valid.any():return pred.sum()*0.
    return (1-(F.normalize(pred[valid],dim=-1,eps=1e-8)*target[valid]).sum(-1)).clamp_min(0.).mean()


class ContextTeacher(nn.Module):
    def __init__(self,encoder,graph,hidden):
        super().__init__()
        # Deepcopy of a PyG FX GraphModule can turn nested ModuleDicts into plain
        # Modules. Copy the named relation layers explicitly, keeping indexing.
        self.teacher=nn.Module()
        for name in ['conv1','conv2']:
            self.teacher.add_module(name,nn.ModuleDict({k:copy.deepcopy(v) for k,v in getattr(encoder.original,name).items()}))
        self.teacher.requires_grad_(False)
        self.tokens=nn.ParameterDict({n:nn.Parameter(graph[n].x.mean(0).clone()) for n in ['miRNA','drug']})
        self.predictor=nn.ModuleDict({n:nn.Sequential(nn.Linear(2*hidden+4,128),nn.GELU(),nn.Linear(128,hidden)) for n in ['miRNA','drug']})
        self.register_buffer('teacher_updates',torch.tensor(0,dtype=torch.long))

    def context(self,encoder,graph,masked,inputs=None):
        source={n:graph[n].x for n in ['miRNA','drug']} if inputs is None else inputs
        damaged={n:torch.where(masked[n][:,None],self.tokens[n],x) for n,x in source.items()}
        states=encode(encoder,encoder.original,damaged)
        states={n:F.layer_norm(h,(h.shape[1],),eps=1e-5) for n,h in states.items()}
        contexts={}
        for node in ['miRNA','drug']:
            # Same-type relation first, target relation second, independent of
            # graph relation insertion order. Self loops count as released.
            edges=sorted([e for e in graph.edge_types if e[2]==node],key=lambda e:e[0]!=node)
            assert len(edges)==2
            values=[];descriptors=[]
            for edge in edges:
                adj=encoder.adj[edge];visible=(~masked[edge[0]]).to(states[edge[0]].dtype)[:,None]
                degree=torch.sparse.mm(adj,torch.ones_like(visible))
                count=torch.sparse.mm(adj,visible)
                value=torch.sparse.mm(adj,states[edge[0]]*visible)/count.clamp_min(1)
                values.append(value);descriptors.extend([degree.log1p(),count.log1p()])
            contexts[node]=torch.cat(values+descriptors,-1)
        return contexts

    def loss(self,encoder,graph,masked):
        self.teacher.eval()
        with torch.no_grad():
            target=encode(encoder,self.teacher,{n:graph[n].x for n in ['miRNA','drug']})
            target={n:normalized_target(h) for n,h in target.items()}
        contexts=self.context(encoder,graph,masked)
        typed={n:cosine_loss(self.predictor[n](contexts[n][masked[n]]),target[n][masked[n]]) for n in ['miRNA','drug']}
        return .5*(typed['miRNA']+typed['drug']),typed

    @torch.no_grad()
    def update_teacher(self,encoder,momentum):
        for (tn,t),(sn,s) in zip(self.teacher.named_parameters(),encoder.original.named_parameters()):
            assert tn==sn
            if momentum==0:t.copy_(s)
            else:t.mul_(momentum).add_(s,alpha=1-momentum)
        for (tn,t),(sn,s) in zip(self.teacher.named_buffers(),encoder.original.named_buffers()):
            assert tn==sn;t.copy_(s)
        self.teacher_updates.add_(1)


def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name!='control':a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    if name in CONFIGS:
        a.model.add_module('context_teacher',ContextTeacher(a.model.encoder,a.graph,a.cfg.hidden).to(device))
        a.context_config=CONFIGS[name]
    return a


def mask_generator(device,seed):return torch.Generator(device=device).manual_seed(seed)


def load_masks(fold,device):
    with np.load(HERE/'masks.npz') as z:
        return {kind:{n:torch.as_tensor(z[f'{kind}_{fold}_{n}'],device=device,dtype=torch.bool) for n in ['miRNA','drug']} for kind in ['block','random']}


def make_optimizer(m):
    return torch.optim.RMSprop([
        dict(params=[p for n,p in m.named_parameters() if not n.startswith(('reconstruction.','context_teacher.'))],lr=1e-4),
        dict(params=m.reconstruction.parameters(),lr=1e-4,weight_decay=1e-4),
        dict(params=[p for p in m.context_teacher.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)],alpha=.99)


def step(a,opt,epoch,generator):
    m=a.model;m.train();m.context_teacher.teacher.eval();opt.zero_grad(set_to_none=True)
    bce=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)
    idx=m.reconstruction.masks(a.graph,.5,generator);rec,_=m.reconstruction.loss(m.encoder,a.graph,idx,False)
    masked={n:x[epoch-1] for n,x in a.mask_schedule[a.context_config['mask']].items()}
    latent,typed=m.context_teacher.loss(m.encoder,a.graph,masked)
    rw=10*min(1.,epoch/50.);lw=.1*min(1.,max(0.,(epoch-30)/30.))
    loss=bce+rw*rec
    if lw>0:loss=loss+lw*latent
    assert torch.isfinite(loss);loss.backward();opt.step()
    assert all(p.grad is None for p in m.context_teacher.teacher.parameters())
    m.context_teacher.update_teacher(m.encoder,a.context_config['momentum']);m.eval()
    return dict(bce=float(bce.detach()),reconstruction=float(rec.detach()),reconstruction_weight=rw,
                latent=float(latent.detach()),latent_miRNA=float(typed['miRNA'].detach()),latent_drug=float(typed['drug'].detach()),
                latent_weight=lw,teacher_updates=int(m.context_teacher.teacher_updates),loss=float(loss.detach()))
