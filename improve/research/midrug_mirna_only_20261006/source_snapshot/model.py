"""Strict miRNA-only M123 with optional masked grouping evidence in M3."""
import torch
from data import HERE,RELEASE,make_graph,load_inner,assert_clean,sha256
from groups import GroupEvidence
from incrd_hg_m123.adapter import Adapter
from incrd_hg_m123.model import Model
from incrd_hg_m123.pair_channels import PairStruct
from incrd_hg_m123.engine import seed_all

CONFIGS=('control','rna','drug','combined')


class GroupPairStruct(PairStruct):
    def __init__(self,D,pos,cfg,evidence,mode):
        super().__init__(D,pos,cfg);self.evidence=evidence;self.mode=mode
    def qis(self,r,d):
        return torch.cat([super().qis(r,d),self.evidence.features(r,d,self.mode)],1)


def make_adapter(data,mode,device):
    assert mode in CONFIGS
    a=Adapter(data,make_graph(data),device);assert_clean(a.graph)
    assert all('lnc' not in name.lower() for name,_ in a.model.named_parameters())
    if mode=='control':return a
    initial=a.model.state_dict();base_width=a.qis['train'].shape[1]
    evidence=GroupEvidence(data,device)
    ps=GroupPairStruct(a.D,a.graph['miRNA','MiDrug','drug'].edge_index.T,a.cfg,evidence,mode)
    a.ps=ps
    raw=ps.qis(*a.pairs['train']);a.mu,a.sd=raw.mean(0,keepdim=True),raw.std(0,keepdim=True)+1e-6
    a.qis={p:(ps.qis(*idx)-a.mu)/a.sd for p,idx in a.pairs.items()}
    a.ease={p:ps.ease(*idx) for p,idx in a.pairs.items()}
    a.dense=(*a.dense[:4],ps.A)
    model=Model(a.graph,ps,raw.shape[1],a.ease['train'].shape[1],a.cfg).to(device)
    original=initial['pc.0.weight']
    initial['pc.0.weight']=torch.cat([original,torch.zeros((original.shape[0],raw.shape[1]-base_width),device=device)],1)
    model.load_state_dict(initial)
    a.model=model
    assert all('lnc' not in name.lower() for name,_ in a.model.named_parameters())
    return a
