"""SCONE-inspired differentiable score sampling for graph training.

The variance-exploding reverse discretization follows paper Algorithms 1/2.
Only latent training augmentations are generated; no graph edge or biological
association is synthesized as observed data. Prediction uses the graph encoder.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import sharpness_graph_v1 as ordinary
from improve import nested_anchor_inputs_v2 as nested
from improve import overlap_block_models as graph_lib

KINDS = ('full','no_injection','no_reverse_drift')
identity_task = ordinary.identity_task


def parameters():
    return dict(width=64,heads=4,depth=2,dropout=.1,folds=5,lnc_policy='identity_strong_reference',
        score_hidden=128,score_lr=.001,score_weight_decay=0.,sigma_min=.01,sigma_max=50.,
        reverse_steps=10,intermediate_time=.1,injection_weight=.9,triplet_batch=256,
        contrastive_weight=.05,ranking_weight=.1,contrastive_temperature=.2,
        score_parameterization='negative_noise_over_sigma',score_time_embedding='sigma',
        reverse_discretization='paper_variance_increments_with_noise_at_every_step')


def sigma_at(time,p):
    return p['sigma_min']*torch.exp(time*math.log(p['sigma_max']/p['sigma_min']))


class ScoreNetwork(nn.Module):
    def __init__(self,width,hidden):
        super().__init__()
        self.width = width
        self.time = nn.Sequential(nn.Linear(width,4*width),nn.Tanh(),nn.Linear(4*width,4*width))
        self.input = nn.Linear(width,width)
        self.encode = nn.Linear(width,hidden)
        self.time_encode = nn.Linear(4*width,hidden)
        self.encode_output = nn.Linear(hidden,hidden)
        self.bottom = nn.Linear(hidden,hidden)
        self.decode = nn.Linear(2*hidden,width)
        self.time_decode = nn.Linear(4*width,width)
        self.decode_output = nn.Linear(width,width)
        self.output = nn.Linear(width,width)

    def forward(self,x,sigma):
        half = self.width//2
        frequencies = torch.exp(torch.arange(half,device=x.device,dtype=x.dtype)*(-math.log(10000.)/max(half-1,1)))
        phases = sigma[:,None]*frequencies[None,:]
        time = self.time(torch.cat((phases.sin(),phases.cos()),-1))
        h0 = self.input(x)
        h1 = self.encode_output(torch.tanh(self.encode(h0))+self.time_encode(time))
        h2 = torch.tanh(self.bottom(h1))
        h3 = self.decode_output(torch.tanh(self.decode(torch.cat((h2,h1),-1)))+self.time_decode(time))
        return self.output(h3)

    def score(self,x,sigma):
        return -self(x,sigma)/sigma[:,None]


class AugmentationGraph(ordinary.OrdinaryGraph):
    def __init__(self,dimensions,relations,p):
        super().__init__(dimensions,relations,p)
        self.score_network = ScoreNetwork(p['width'],p['score_hidden'])

    def encode(self,xs,graph):
        edges,relation,counts = graph
        parts,start = [],0
        for x,project in zip(xs,self.project):
            rms = x.square().mean(1,keepdim=True).clamp_min(1e-8).sqrt()
            parts.append(project(torch.cat((x/rms,counts[start:start+len(x)]),-1)))
            start += len(x)
        h = F.gelu(self.norm(torch.cat(parts)))
        for layer in self.layers:
            h = layer(h,edges,relation)
        return h

    def decode(self,left,right):
        return self.readout(torch.cat((left,right,left*right,(left-right).abs()),-1)).squeeze(-1)

    def forward(self,xs,graph,pairs):
        h = self.encode(xs,graph)
        return self.decode(h[pairs[0]],h[pairs[1]])


def setup(a,p,kind,device):
    if kind not in KINDS or set(a)-(nested.TRAIN_KEYS|{'anchor_oof'}):
        raise ValueError('Unregistered arm or non-training graph inputs')
    folds = [torch.as_tensor(np.flatnonzero(a['folds']==f),device=device) for f in range(p['folds'])]
    if identity_task(a,p):
        return ordinary.IdentityResidual().to(device),[],[None]*p['folds'],folds
    n,relations = int(a['node_counts'].sum()),int(a['relation'].max())+1
    graphs = []
    for f in range(p['folds']):
        edges,labels = graph_lib.masked_physical(a,np.flatnonzero(a['folds']==f))
        graphs.append(graph_lib.graph_inputs(edges,labels,np.empty((2,0),dtype=np.int64),n,relations,device))
    xs = [torch.as_tensor(a['x_'+str(i)],device=device) for i in range(len(a['node_counts']))]
    model = AugmentationGraph([x.shape[1] for x in xs],relations,p).to(device)
    return model,xs,graphs,folds


def reverse_sample(score_fn,features,p,noise,kind,inject=False):
    """Matched finite-step VE sampling; noise is explicit for replay/controls."""
    steps = p['reverse_steps']
    if kind not in KINDS or noise.shape != (steps+1,*features.shape):
        raise ValueError('Unexpected sampler inputs')
    if inject and len(features)%2:
        raise ValueError('Injection requires paired positive and negative embeddings')
    times = torch.linspace(p['intermediate_time'],0.,steps+1,device=features.device,dtype=features.dtype)
    sigmas = sigma_at(times,p)
    x = features + (sigmas[0].square()-sigmas[-1].square()).sqrt()*noise[0]
    initial = x
    weight = 1. if kind == 'no_injection' else p['injection_weight']
    drift_multiplier = 0. if kind == 'no_reverse_drift' else 1.
    for i in range(steps):
        if inject:
            pos,neg = x.chunk(2,dim=0)
            x = torch.cat((pos,weight*neg+(1-weight)*pos),0)
        delta = sigmas[i].square()-sigmas[i+1].square()
        score = score_fn(x,sigmas[i].expand(len(x)))
        x = x + drift_multiplier*delta*score + delta.sqrt()*noise[i+1]
    return initial,x


def denoising_loss(model,features,p,time,noise):
    detached = features.detach()
    sigma = sigma_at(time,p)
    noisy = detached+sigma[:,None]*noise
    predicted_noise = model(noisy,sigma)
    return (predicted_noise-noise).square().mean()


def contrastive(first,second,temperature):
    first,second = F.normalize(first,dim=-1),F.normalize(second,dim=-1)
    logits = first @ second.T/temperature
    labels = torch.arange(len(first),device=first.device)
    return .5*(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))


def training_triplets(a,device,fold_count=5):
    """Use only actually labelled query-fold positives and zero-labelled pairs."""
    pools = []
    for fold in range(fold_count):
        ids = np.flatnonzero(a['folds']==fold)
        negative = {}
        for row in ids[a['y_train'][ids]==0]:
            negative.setdefault(int(a['pairs_train'][0,row]),[]).append(int(row))
        positive = [int(row) for row in ids[a['y_train'][ids]==1]
                    if int(a['pairs_train'][0,row]) in negative]
        starts,lengths,flat = [],[],[]
        for row in positive:
            matches = negative[int(a['pairs_train'][0,row])]
            starts.append(len(flat)); lengths.append(len(matches)); flat.extend(matches)
        if not positive:
            raise ValueError('An omitted fold lacks paired positive/negative training examples')
        pools.append(tuple(torch.as_tensor(x,device=device,dtype=torch.long) for x in (positive,starts,lengths,flat)))
    return pools


def sample_triplets(pool,count,generator):
    positive,starts,lengths,flat = pool
    sampled = torch.randint(len(positive),(count,),device=positive.device,generator=generator)
    uniform = torch.rand(count,device=positive.device,generator=generator)
    within = torch.floor(uniform*lengths[sampled]).long()
    return positive[sampled],flat[starts[sampled]+within]
