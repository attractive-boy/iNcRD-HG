"""Matched pair heads for an existing structural HGT and its MLP control."""
import copy
import hashlib
import random

import numpy as np
import torch
from torch import nn

from improve.ragt_modules import StructuralHGT
from improve.pair_exchange_modules import remove_target_pairs, sample_queries


PAIR_DIM=62


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def pair_head(width,dropout):
    return nn.Sequential(nn.Linear(PAIR_DIM,width),nn.LeakyReLU(.1),nn.Dropout(dropout),
                         nn.Linear(width,width),nn.LeakyReLU(.1),nn.Linear(width,1))


class MatchedMLP(nn.Module):
    def __init__(self,head):
        super().__init__();self.feature_mlp=head

    def forward(self,x_dict,edges,pairs,features):
        return self.feature_mlp(features).view(-1)


def make_model(architecture,cfg,x_dict,edges,rna_type,seed,device):
    if architecture not in ('mlp','sahgt'):
        raise ValueError('Unregistered trainable architecture')
    if cfg['fusion']!='add':
        raise ValueError('Matched pair-head comparison requires additive fusion')
    set_seed(seed)
    if architecture=='sahgt':
        model=StructuralHGT({nt:x.shape[1] for nt,x in x_dict.items()},list(edges),rna_type,PAIR_DIM,
            width=cfg['width'],depth=cfg['depth'],heads=cfg['heads'],dropout=cfg['dropout'],
            fusion='add',gate_init=cfg['gate_init']).to(device)
        # Materialize lazy node projections without using labels or pair features.
        model.eval()
        with torch.no_grad():
            model(x_dict,edges,torch.zeros((2,2),dtype=torch.long,device=device),
                  torch.zeros((2,PAIR_DIM),device=device))
    # Both models and all input conditions start with exactly the same pair head.
    set_seed(seed+20260913)
    head=pair_head(cfg['width'],cfg['dropout']).to(device)
    if architecture=='mlp':
        model=MatchedMLP(head).to(device)
    else:
        model.feature_mlp=copy.deepcopy(head)
    set_seed(seed+20260914)
    return model


def training_context(pairs,labels,edges,weights,task,rna_type,n_drug,fraction,seed,epoch):
    """Use one deterministic query schedule and hide each query positive."""
    idx=sample_queries(labels,fraction,(seed+1)*1000003+epoch)
    held=pairs[:,idx[labels[idx]==1]]
    visible,_=remove_target_pairs(edges,weights,task,rna_type,n_drug,held)
    return idx,visible


def state_digest(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        arr=value.detach().cpu().contiguous().numpy()
        h.update(key.encode());h.update(str(arr.shape).encode());h.update(str(arr.dtype).encode());h.update(arr.tobytes())
    return h.hexdigest()


@torch.no_grad()
def predict(model,x_dict,edges,pairs,features):
    """Raw logits preserve ranking when float32 sigmoid would saturate."""
    model.eval()
    score=model(x_dict,edges,pairs,features).cpu().numpy().astype(np.float64)
    if score.shape!=(pairs.shape[1],) or not np.isfinite(score).all():
        raise ValueError('Invalid model prediction')
    return score


@torch.no_grad()
def graph_contribution(model,x_dict,edges,pairs,features):
    if isinstance(model,MatchedMLP):
        return dict(graph_gate=None,graph_to_pair_logit_std_ratio=None)
    model.eval();z=model.encode(x_dict,edges)
    r,d=pairs;zr,zd=z[model.rna_type][r],z['drug'][d]
    pair=model.feature_mlp(features).ravel()
    graph=model.graph_gate.sigmoid()*model.graph_mlp(torch.cat([zr,zd,(zr-zd).abs(),zr*zd],-1)).ravel()
    return dict(graph_gate=float(model.graph_gate.sigmoid().cpu()),
                graph_to_pair_logit_std_ratio=float((graph.std()/pair.std().clamp_min(1e-12)).cpu()))
