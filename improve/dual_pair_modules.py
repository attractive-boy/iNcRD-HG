"""Parallel local-interaction and transported-evidence pair states.

This follow-up preserves a full pointwise nonlinear path alongside pair-axis
transport. Equal-size controls make both paths pointwise or both transported.
All paths share node features and initial pair projections and train jointly.
"""
import torch
from torch import nn

from improve.pair_state_modules import PairBlock, PairStateBranch, PairStateModel


class DualPairBranch(PairStateBranch):
    def __init__(self, graph, hidden, rna_type, mode='dual_pair', width=16, depth=2):
        if mode not in ('dual_local','dual_transport','dual_pair'):
            raise ValueError(mode)
        super().__init__(graph, hidden, rna_type, mode='pair_features', width=width, depth=depth)
        self.arch_mode = mode
        # The first experiment did not support an explicit association token.
        # Existing support edges still enter the unchanged node backbone.
        del self.boundary_vectors
        self.local_blocks = nn.ModuleList([PairBlock(self.types,self.relations,width) for _ in range(depth)])
        self.head = nn.Sequential(nn.Linear((2*depth+1)*width,2*width),
                                  nn.LeakyReLU(.1),nn.Linear(2*width,1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, z, edge_index_dict, pairs, intervention=None):
        if intervention not in (None,'no_transport','zero_local'):
            raise ValueError(intervention)
        drug_main,drug_product=self.drug_projection(z['drug']).chunk(2,dim=-1)
        initial={}
        for t in self.types:
            main,product=self.rna_projections[t](z[t]).chunk(2,dim=-1)
            initial[t]=(main[:,None,:]+drug_main[None,:,:]
                        +product[:,None,:]*drug_product[None,:,:]).tanh()
        local,transported=initial,initial
        local_moves=self.arch_mode=='dual_transport'
        other_moves=self.arch_mode!='dual_local'
        if intervention=='no_transport':
            local_moves=other_moves=False
        r,d=pairs
        selected=[initial[self.rna_type][r,d]]
        for a,b in zip(self.local_blocks,self.blocks):
            local=a(local,self.rna_operators,self.drug_operator,transport=local_moves)
            transported=b(transported,self.rna_operators,self.drug_operator,transport=other_moves)
            own=local[self.rna_type][r,d]
            selected.extend([torch.zeros_like(own) if intervention=='zero_local' else own,
                             transported[self.rna_type][r,d]])
        return self.head(torch.cat(selected,dim=-1)).flatten()


class DualPairModel(PairStateModel):
    def __init__(self,base,graph,hidden,rna_type,arm,width=16,depth=2):
        nn.Module.__init__(self)
        self.base=base
        self.branch=(None if arm in ('pca_query','pca_default') else
                     DualPairBranch(graph,hidden,rna_type,arm,width,depth))
