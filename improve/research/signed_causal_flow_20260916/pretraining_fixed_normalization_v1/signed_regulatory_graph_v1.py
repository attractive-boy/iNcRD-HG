"""Bounded, sign-constrained regulatory mediation of MiDrug representations.

Curated protein regulation is a separate evidence channel. Annotation links
are feature probes: neither drug binding nor RNA binding is assigned an
inhibition label. This is an independent finite-depth construction, not GSNN
code or an identified cellular causal model.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import nested_anchor_inputs_v2 as nested
from improve import overlap_block_models as graph_lib
from improve import tabular_foundation as base
from improve import signed_regulatory_inputs_v1 as inputs
from improve.sharpness_graph_v1 import IdentityResidual, identity_task

KINDS = ('full', 'unsigned', 'shuffled_sign')
TRAIN_KEYS = nested.TRAIN_KEYS | {'anchor_oof'}


def parameters():
    source = inputs.OUT/'arrays.npz'
    manifest = inputs.OUT/'DATA_AUDIT.json'
    return dict(width=64, heads=4, depth=2, dropout=.1, folds=5,
        gene_width=16, regulatory_steps=8, coupling=.4, maximum_gain=2.,
        regulatory_operator='directed signed incoming-degree-normalized curated edges; no reverse augmentation',
        regulatory_dynamics='eight tanh updates with positive gene/channel gains in (0,2); finite iteration, not a converged solver',
        binding_direction='unknown; positive annotation matrices are feature probes only',
        evidence_arrays=str(source), evidence_sha256=base.digest(source),
        evidence_manifest=str(manifest), evidence_manifest_sha256=base.digest(manifest),
        lnc_policy='identity_strong_reference')


def evidence(p):
    path = Path(p['evidence_arrays'])
    if base.digest(path) != p['evidence_sha256']:
        raise ValueError('Regulatory evidence arrays changed')
    manifest = Path(p['evidence_manifest'])
    if base.digest(manifest) != p['evidence_manifest_sha256']:
        raise ValueError('Regulatory evidence manifest changed')
    record = base.check_seal(base.read_json(manifest))
    if record['association_labels_read'] or not record['no_binding_action_direction_inferred']:
        raise ValueError('Invalid regulatory evidence semantics')
    for file,digest in record.get('source_sha256',{}).items():
        if base.digest(file) != digest: raise ValueError('Evidence builder changed')
    for file,digest in record.get('input_sha256',{}).items():
        if base.digest(file) != digest: raise ValueError('Evidence source changed')
    with np.load(path, allow_pickle=False) as z:
        return {key:z[key].copy() for key in z.files}


def normalized_incidence(indices, values, shape, device):
    """Separate row-mean pooling and column-mean feature injection."""
    index = torch.as_tensor(indices, device=device)
    value = torch.as_tensor(values, device=device)
    if torch.any(value <= 0) or not torch.isfinite(value).all():
        raise ValueError('Binding weights must be finite and positive')
    row_mass = value.new_zeros(int(shape[0])).index_add(0,index[0],value)
    col_mass = value.new_zeros(int(shape[1])).index_add(0,index[1],value)
    pooling = torch.sparse_coo_tensor(index,value/row_mass[index[0]],tuple(map(int,shape))).coalesce()
    injection = torch.sparse_coo_tensor(index.flip(0),value/col_mass[index[1]],
                                        tuple(map(int,shape[::-1]))).coalesce()
    return pooling,injection,row_mass,col_mass


def regulatory_operator(edges, signs, nodes, device):
    edge = torch.as_tensor(edges,device=device)
    sign = torch.as_tensor(signs,device=device)
    if not torch.all((sign == 1) | (sign == -1)):
        raise ValueError('Only explicit binary regulatory signs are allowed')
    if edge.shape != (2,len(sign)):
        raise ValueError('Invalid regulatory edge shape')
    incoming = torch.bincount(edge[1],minlength=nodes).to(sign.dtype)
    return torch.sparse_coo_tensor(edge.flip(0),sign/incoming[edge[1]],(nodes,nodes)).coalesce()


def regulatory_response(operator, forcing, raw_gain, bias, steps, coupling=.4, maximum_gain=2.):
    """Finite iterates of a contractive, sign-constrained nonlinear map.

    Since ||operator||_infinity <= 1 and tanh is 1-Lipschitz, the derivative
    wrt the previous state has norm at most coupling*maximum_gain < 1.
    This bound does not assert eight updates reach a fixed-point tolerance.
    """
    if not 0 <= coupling*maximum_gain < 1 or steps < 1:
        raise ValueError('A strict contraction bound and positive depth are required')
    gain = maximum_gain*raw_gain.sigmoid()
    state = forcing.tanh()
    for _ in range(steps):
        state = (gain*(forcing+coupling*torch.sparse.mm(operator,state))+bias).tanh()
    return state


class SignedRegulatoryGraph(nn.Module):
    def __init__(self, dimensions, node_counts, relations, p, kind, data, device):
        super().__init__()
        if kind not in KINDS: raise ValueError('Unregistered regulatory mechanism')
        self.p,self.kind = dict(p),kind
        width,gw = p['width'],p['gene_width']
        if int(data['rna_shape'][0]) != int(node_counts[1]) or int(data['drug_shape'][0]) != int(node_counts[2]):
            raise ValueError('Regulatory annotation entity count mismatch')
        self.project = nn.ModuleList([nn.Linear(d+relations,width) for d in dimensions])
        self.norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([graph_lib.SparseTypedAttention(width,p['heads'],
            2*relations+2,p['dropout']) for _ in range(p['depth'])])
        self.rna_probe,self.drug_probe = nn.Linear(width,gw),nn.Linear(width,gw)
        genes = len(data['genes'])
        self.raw_gene_gain = nn.Parameter(torch.zeros(genes,gw))
        self.gene_bias = nn.Parameter(torch.zeros(genes,gw))
        self.rna_bio_norm,self.drug_bio_norm = nn.LayerNorm(gw),nn.LayerNorm(gw)
        for prefix in ('rna','drug'):
            pool,inject,row_mass,col_mass = normalized_incidence(data[prefix+'_indices'],
                data[prefix+'_values'],data[prefix+'_shape'],device)
            for label,value in (('pool',pool),('inject',inject),('mass',row_mass),('gene_mass',col_mass)):
                self.register_buffer(prefix+'_'+label,value)
        signs = data['regulatory_sign'] if kind == 'full' else (
            np.ones_like(data['regulatory_sign']) if kind == 'unsigned' else data['shuffled_sign'])
        self.register_buffer('regulatory',regulatory_operator(data['regulatory_edges'],signs,genes,device))
        self.readout = nn.Sequential(nn.Linear(4*width+4*gw+4,width),nn.GELU(),
            nn.Dropout(p['dropout']),nn.Linear(width,1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.rna_degree,self.drug_degree = nn.Identity(),nn.Identity()
        starts=np.r_[0,np.cumsum(node_counts)]
        self.rs,self.re,self.ds,self.de = map(int,(starts[1],starts[2],starts[2],starts[3]))
        self.regulatory_diagnostics = dict(forwards=0,regulatory_updates=0,
            maximum_absolute_state=0.,maximum_gain=0.)

    def encode(self,xs,graph):
        edges,relation,counts=graph
        parts,start=[],0
        for x,projection in zip(xs,self.project):
            rms=x.square().mean(1,keepdim=True).clamp_min(1e-8).sqrt()
            parts.append(projection(torch.cat((x/rms,counts[start:start+len(x)]),-1)))
            start+=len(x)
        h=F.gelu(self.norm(torch.cat(parts)))
        for layer in self.layers: h=layer(h,edges,relation)
        r=self.rna_probe(h[self.rs:self.re]).tanh()
        d=self.drug_probe(h[self.ds:self.de]).tanh()
        # Both roles have a fixed half weight; missing roles contribute zero.
        forcing=.5*(torch.sparse.mm(self.rna_inject,r)+torch.sparse.mm(self.drug_inject,d))
        state=regulatory_response(self.regulatory,forcing,self.raw_gene_gain,self.gene_bias,
            self.p['regulatory_steps'],self.p['coupling'],self.p['maximum_gain'])
        if not torch.isfinite(state).all(): raise FloatingPointError('Nonfinite regulatory state')
        rb=self.rna_bio_norm(torch.sparse.mm(self.rna_pool,state))
        db=self.drug_bio_norm(torch.sparse.mm(self.drug_pool,state))
        self.regulatory_diagnostics['regulatory_updates']+=self.p['regulatory_steps']
        self.regulatory_diagnostics['maximum_absolute_state']=max(
            self.regulatory_diagnostics['maximum_absolute_state'],float(state.detach().abs().max()))
        self.regulatory_diagnostics['maximum_gain']=max(self.regulatory_diagnostics['maximum_gain'],
            float((self.p['maximum_gain']*self.raw_gene_gain.detach().sigmoid()).max()))
        return h,rb,db

    def forward(self,xs,graph,pairs):
        r,d=pairs
        if torch.any((r < self.rs)|(r >= self.re)) or torch.any((d < self.ds)|(d >= self.de)):
            raise ValueError('Regulatory decoder expects MiDrug pairs')
        h,rb,db=self.encode(xs,graph)
        ri,di=r-self.rs,d-self.ds
        left,right=h[r],h[d]
        br,bd=rb[ri],db[di]
        support=torch.stack((self.rna_mass[ri].log1p(),self.drug_mass[di].log1p(),
            (self.rna_mass[ri]>0).to(h.dtype),(self.drug_mass[di]>0).to(h.dtype)),-1)
        features=torch.cat((left,right,left*right,(left-right).abs(),br,bd,br*bd,(br-bd).abs(),support),-1)
        self.regulatory_diagnostics['forwards']+=1
        return self.readout(features).squeeze(-1)


def setup(a,p,kind,device):
    if set(a)-TRAIN_KEYS: raise ValueError('Regulatory setup accepts training arrays only')
    if kind not in KINDS: raise ValueError('Unregistered regulatory mechanism')
    folds=[torch.as_tensor(np.flatnonzero(a['folds']==f),device=device) for f in range(p['folds'])]
    if identity_task(a,p):
        model=IdentityResidual().to(device)
        model.regulatory_diagnostics={}
        return model,[],[None]*p['folds'],folds
    data=evidence(p)
    n,relations=int(a['node_counts'].sum()),int(a['relation'].max())+1
    graphs=[]
    for fold in range(p['folds']):
        e,labels=graph_lib.masked_physical(a,np.flatnonzero(a['folds']==fold))
        graphs.append(graph_lib.graph_inputs(e,labels,np.empty((2,0),dtype=np.int64),n,relations,device))
    xs=[torch.as_tensor(a['x_'+str(i)],device=device) for i in range(len(a['node_counts']))]
    model=SignedRegulatoryGraph([x.shape[1] for x in xs],a['node_counts'],relations,p,kind,data,device).to(device)
    return model,xs,graphs,folds
