"""Five exact-query-excluded name-neighbor supports enter the existing M3 readout."""
import json
import numpy as np
import torch
from common import HERE, Adapter, Reconstruction, make_graph, sha256
from incrd_hg_m123.model import Model
from incrd_hg_m123.pair_channels import PairStruct

CONFIGS={name:dict(kernel=name,reconstruction_weight=10.) for name in ['aligned','permuted','zero']}

class NameEvidence:
    def __init__(self,data,name,device):
        meta=json.loads((HERE/'input_manifest.json').read_text())
        assert sha256(HERE/'covariates.npz')==meta['covariates_sha256']
        with np.load(HERE/'covariates.npz') as z:self.kernel=torch.as_tensor(z[name],device=device)
        assert not torch.count_nonzero(self.kernel.diag())
        pos=torch.as_tensor(data['train_pairs'][data['train_y']==1],device=device)
        neg=torch.as_tensor(data['train_pairs'][data['train_y']==0],device=device)
        A=torch.zeros((605,216),dtype=torch.float64,device=device);N=torch.zeros_like(A)
        A[pos[:,0],pos[:,1]]=1.;N[neg[:,0],neg[:,1]]=1.
        P,Q=A@self.kernel,N@self.kernel
        mass=self.kernel.sum(0)[None,:]
        self.values=torch.stack([torch.log1p(P),torch.log1p(Q),P/(P+Q+1.),
                                 torch.log1p(P+Q),P/(mass+1.)],dim=-1).float()
    def features(self,r,d):return self.values[r,d]

class NamePairStruct(PairStruct):
    def __init__(self,D,pos,cfg,evidence):
        super().__init__(D,pos,cfg);self.evidence=evidence
    def qis(self,r,d):return torch.cat([super().qis(r,d),self.evidence.features(r,d)],dim=1)

def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    if name=='baseline':return a
    assert name in CONFIGS
    initial=a.model.state_dict();base_width=a.qis['train'].shape[1]
    evidence=NameEvidence(data,name,device)
    ps=NamePairStruct(a.D,a.graph['miRNA','MiDrug','drug'].edge_index.T,a.cfg,evidence)
    a.ps=ps
    raw=ps.qis(*a.pairs['train']);a.mu,a.sd=raw.mean(0,keepdim=True),raw.std(0,keepdim=True)+1e-6
    a.qis={part:(ps.qis(*idx)-a.mu)/a.sd for part,idx in a.pairs.items()}
    a.ease={part:ps.ease(*idx) for part,idx in a.pairs.items()}
    a.dense=(*a.dense[:4],ps.A)
    m=Model(a.graph,ps,raw.shape[1],a.ease['train'].shape[1],a.cfg).to(device)
    m.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    weight=initial['pc.0.weight']
    initial['pc.0.weight']=torch.cat([weight,weight.new_zeros(weight.shape[0],raw.shape[1]-base_width)],dim=1)
    m.load_state_dict(initial);a.model=m
    return a

def mask_generator(device,fold):return torch.Generator(device=device).manual_seed(20261006+fold)
