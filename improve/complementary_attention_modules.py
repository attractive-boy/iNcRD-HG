"""Residual channel/path experts with evidence-dependent routing and abstention."""
import math

import numpy as np
import torch
from torch import nn

from improve.attention_mechanisms import ChannelBranch, norm
from improve.drug_condition_attention import masked_weights, set_seed
from improve.smooth_path_attention import SmoothPathBranch, SmoothPathModel


ARMS = ('base', 'channel_mlp', 'channel_attention', 'path_attention',
        'fusion_fixed', 'fusion_global', 'fusion_gate', 'fusion_gate_simple', 'fusion_no_null')
PRIMARY = ('fusion_gate',)
FUSIONS = tuple(arm for arm in ARMS if arm.startswith('fusion_'))


def routing_assets(train, paths, rna_type, task):
    """Label-free covariates; remove the queried association from degree counts."""
    key = (rna_type, task, 'drug')
    edges = set(map(tuple, train[key].edge_index.cpu().t().tolist()))
    nr, nd = train[rna_type].num_nodes, train['drug'].num_nodes
    degree_r, degree_d = np.zeros(nr), np.zeros(nd)
    for rna, drug in edges:
        degree_r[rna] += 1
        degree_d[drug] += 1
    result = {}
    for fold, pairs in paths['pairs'].items():
        rna, drug = pairs.numpy()
        membership = np.fromiter(((int(r), int(d)) in edges for r, d in zip(rna, drug)), dtype=np.float32)
        counts = torch.stack([pack['counts'] for pack in paths['packs'][fold]], 1)
        degree = torch.from_numpy(np.stack([degree_r[rna] - membership,
                                           degree_d[drug] - membership], -1)).float()
        features = torch.cat([degree.log1p(), counts.log1p(), (counts > 0).float()], -1)
        available = torch.ones(len(rna), 3, dtype=torch.bool)
        available[:, 2] = (counts > 0).any(1)
        result[fold] = dict(features=features, available=available,
                            query_in_graph=torch.from_numpy(membership).bool())
    return result


class ExpertRouter(nn.Module):
    def __init__(self, hidden, n_evidence, width=16):
        super().__init__()
        self.query_r = nn.Linear(hidden, width, bias=False)
        self.query_d = nn.Linear(hidden, width, bias=False)
        self.evidence = nn.Linear(n_evidence, width)
        self.keys = nn.Linear(width, 3, bias=True)
        nn.init.zeros_(self.keys.weight)
        nn.init.zeros_(self.keys.bias)
        self.width = width

    def forward(self, zr, zd, evidence):
        query = torch.tanh(self.query_r(norm(zr)) + self.query_d(norm(zd)) + self.evidence(evidence))
        return self.keys(query) / math.sqrt(self.width)


class ComplementaryModel(nn.Module):
    def __init__(self, base, paths, routing, config, arm, seed, width=32,
                 branch_rng_offset=3000003, router_rng_offset=5000011):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.base, self.arm, self.rna_type = base, arm, config['rna']
        self.channel, self.path, self.router, self.global_logits = None, None, None, None
        self.last_attention = {}
        for fold, pairs in paths['pairs'].items():
            self.register_buffer('pairs_' + fold, pairs.clone())
            if arm in FUSIONS:
                for name in ('features', 'available'):
                    self.register_buffer(name + '_' + fold, routing[fold][name].clone())
        # Independently seeded experts start identically alone and in a fusion.
        if arm.startswith('channel_') or arm in FUSIONS:
            set_seed(seed + branch_rng_offset)
            mode = 'channel_mlp' if arm in ('channel_mlp', 'fusion_gate_simple') else 'channel_attention'
            self.channel = ChannelBranch(config['hidden'], (self.rna_type, 'drug'), mode)
        if arm == 'path_attention' or arm in FUSIONS:
            set_seed(seed + branch_rng_offset)
            mode = 'path_uniform' if arm == 'fusion_gate_simple' else 'temperature4'
            self.path = SmoothPathBranch(config['hidden'], self.rna_type, paths, mode, width)
        if arm in ('fusion_gate', 'fusion_gate_simple', 'fusion_no_null'):
            set_seed(seed + router_rng_offset)
            self.router = ExpertRouter(config['hidden'], routing['train']['features'].size(1))
        if arm == 'fusion_global':
            self.global_logits = nn.Parameter(torch.zeros(3))

    def forward(self, x, edges, fold, intervention=None):
        z = self.base.encoder(x, edges)
        pairs = getattr(self, 'pairs_' + fold)
        rna, drug = pairs
        base_score = self.base.decoder(z, pairs)
        self.last_attention = {}
        if self.arm == 'base' or intervention == 'base_only':
            return base_score
        channel_score, path_delta = base_score, torch.zeros_like(base_score)
        if self.channel is not None:
            changed, weights = self.channel(None, None, z)
            if intervention == 'channel_mean' and self.channel.mode == 'channel_attention':
                changed = dict(z)
                for nt in (self.rna_type, 'drug'):
                    changed[nt] = z[nt] * weights['channel_' + nt].mean(-1, keepdim=True)
            channel_score = self.base.decoder(changed, pairs)
            self.last_attention.update({name: value.detach() for name, value in weights.items()})
        if self.path is not None:
            path_mode = 'uniform' if intervention == 'path_uniform' else None
            path_delta = self.path(z, pairs, fold, path_mode)
            self.last_attention.update(self.path.last_weights)
        if self.arm.startswith('channel_'):
            return channel_score
        if self.arm == 'path_attention':
            return base_score + path_delta
        available = getattr(self, 'available_' + fold).clone()
        if self.arm == 'fusion_no_null':
            available[:, 0] = False
        if intervention == 'no_path':
            available[:, 2] = False
        if intervention == 'no_channel':
            available[:, 1] = False
        if self.router is not None and intervention != 'fixed_gate':
            logits = self.router(z[self.rna_type][rna], z['drug'][drug], getattr(self, 'features_' + fold))
        elif self.global_logits is not None and intervention != 'fixed_gate':
            logits = self.global_logits.expand(len(rna), -1)
        else:
            logits = base_score.new_zeros((len(rna), 3))
        weights = masked_weights(logits, available)
        self.last_attention['router'] = weights.detach()
        # The null expert has zero residual. This preserves the base score's
        # scale and permits rejection instead of forcing either correction.
        return base_score + weights[:, 1] * (channel_score - base_score) + weights[:, 2] * path_delta

    def attention(self):
        return {name: value.cpu().numpy() for name, value in self.last_attention.items()}


def interventions(arm):
    if arm == 'fusion_gate':
        return ('base_only', 'fixed_gate', 'no_path', 'no_channel', 'channel_mean', 'path_uniform')
    if arm in FUSIONS:
        return ('base_only', 'fixed_gate')
    if arm == 'channel_attention':
        return ('base_only', 'channel_mean')
    if arm == 'path_attention':
        return ('base_only', 'path_uniform')
    return ()
