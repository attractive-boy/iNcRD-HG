"""Typed sum-of-compatible-squares relational density with exact marginals.

Independent finite CP adaptation of Loconte et al. (NeurIPS 2023; AAAI 2025).
Entity tables are learned transductively. The sampled-zero relation is an
observation label, not a biological negative interaction or prevalence model.
"""
import numpy as np
import torch
from torch import nn

from improve import nested_anchor_inputs_v2 as nested
from improve import overlap_block_models as graph_lib
from improve.sharpness_graph_v1 import IdentityResidual, identity_task

KINDS = ('full', 'single_square', 'monotone', 'no_joint')
TRAIN_KEYS = nested.TRAIN_KEYS | {'anchor_oof'}


def parameters():
    return dict(components=4, component_rank=16, initialization_std=.1,
        uniform_mixture=.01, generative_weight=.1, folds=5,
        density_dtype='float64', entity_parameterization='separate subject/object ID tables',
        auxiliary_likelihood='equal average of exact typed conditional log densities for support relations',
        target_zero_relation='sampled unobserved MiDrug association label, support folds only',
        lnc_policy='identity_strong_reference')


def nonnegative_mass(subject, predicate, obj, kind):
    """All inputs have the same batch/component/rank shape after broadcasting."""
    product = subject*predicate*obj
    if kind == 'single_square':
        return product.sum((-2, -1)).square()
    if kind == 'monotone':
        return product.square().sum((-2, -1))
    if kind in ('full', 'no_joint'):
        return product.sum(-1).square().sum(-1)
    raise ValueError('Unregistered circuit arm')


def marginal_mass(subject, predicate, obj, kind):
    """Sum over every subject/object pair without materializing that product."""
    if kind == 'monotone':
        return (subject.square().sum(0)*predicate.square()*obj.square().sum(0)).sum()
    if kind == 'single_square':
        s, o, w = subject.flatten(1), obj.flatten(1), predicate.flatten()
        gs, go = s.T@s, o.T@o
        return (gs*go*w[:, None]*w[None, :]).sum()
    if kind in ('full', 'no_joint'):
        gs = torch.einsum('nmk,nml->mkl', subject, subject)
        go = torch.einsum('nmk,nml->mkl', obj, obj)
        return (gs*go*predicate[..., :, None]*predicate[..., None, :]).sum()
    raise ValueError('Unregistered circuit arm')


def relation_schema(a):
    boundaries = np.r_[0, np.cumsum(a['node_counts'])]
    node_type = np.repeat(np.arange(len(a['node_counts'])), a['node_counts'])
    relations = int(a['relation'].max())+1
    schema = []
    for relation in range(relations):
        e = a['edges'][:, a['relation'] == relation]
        if not e.shape[1]: raise ValueError('Cannot infer an absent relation schema')
        types = node_type[e]
        if np.any(types != types[:, :1]): raise ValueError('Relation crosses inconsistent typed domains')
        schema.append(tuple(map(int, types[:, 0])))
    target = np.unique(a['relation'][a['target_codes'] >= 0])
    if len(target) != 1 or schema[int(target[0])] != (1, 2):
        raise ValueError('Expected a unique miRNA-to-drug target relation')
    return np.asarray(schema+[(1, 2)], dtype=np.int64), int(target[0]), boundaries


class SumSquaresCircuit(nn.Module):
    def __init__(self, counts, schema, target_relation, p, kind):
        super().__init__()
        if kind not in KINDS: raise ValueError('Unregistered circuit arm')
        self.p, self.kind = dict(p), kind
        self.counts = list(map(int, counts))
        self.offsets = np.r_[0, np.cumsum(counts)].tolist()
        self.schema = [tuple(map(int, row)) for row in schema]
        self.target_relation, self.zero_relation = int(target_relation), len(schema)-1
        n, m, k = sum(self.counts), p['components'], p['component_rank']
        self.subject = nn.Parameter(torch.empty(n, m, k))
        self.object = nn.Parameter(torch.empty(n, m, k))
        self.predicate = nn.Parameter(torch.empty(len(schema), m, k))
        for value in (self.subject, self.object, self.predicate):
            nn.init.normal_(value, std=p['initialization_std'])
        self.slope = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()
        self.circuit_diagnostics = dict(forwards=0, marginal_evaluations=0,
            auxiliary_calls=0, auxiliary_facts=0, minimum_normalizer=float('inf'),
            maximum_abs_log_density=0.)

    def factors(self, relation):
        left_type, right_type = self.schema[relation]
        left = self.subject[self.offsets[left_type]:self.offsets[left_type+1]].double()
        right = self.object[self.offsets[right_type]:self.offsets[right_type+1]].double()
        return left, self.predicate[relation].double(), right

    def log_density(self, pairs, relation):
        left_type, right_type = self.schema[relation]
        if torch.any(pairs[0] < self.offsets[left_type]) or torch.any(pairs[0] >= self.offsets[left_type+1]) \
           or torch.any(pairs[1] < self.offsets[right_type]) or torch.any(pairs[1] >= self.offsets[right_type+1]):
            raise ValueError('Pair violates the declared relation schema')
        left, weight, right = self.factors(relation)
        z = marginal_mass(left, weight, right, self.kind)
        if not torch.isfinite(z) or z <= 0:
            raise FloatingPointError('Circuit normalizer is nonfinite or nonpositive')
        f = nonnegative_mass(self.subject[pairs[0]].double(), weight,
                             self.object[pairs[1]].double(), self.kind)
        n = self.counts[left_type]*self.counts[right_type]
        eps = self.p['uniform_mixture']
        probability = (1-eps)*f/z+eps/n
        value = probability.log()
        if not torch.isfinite(value).all(): raise FloatingPointError('Nonfinite circuit log density')
        diag = self.circuit_diagnostics
        diag['marginal_evaluations'] += 1
        diag['minimum_normalizer'] = min(diag['minimum_normalizer'], float(z.detach()))
        diag['maximum_abs_log_density'] = max(diag['maximum_abs_log_density'], float(value.detach().abs().max()))
        return value

    def forward(self, xs, graph, pairs):
        # Parameters, rather than messages containing the query outcome, encode
        # the observations learned so far. No validation labels are inputs.
        positive = self.log_density(pairs, self.target_relation)
        zero = self.log_density(pairs, self.zero_relation)
        self.circuit_diagnostics['forwards'] += 1
        return self.bias.double()+self.slope.double()*(positive-zero)

    def auxiliary_nll(self, graph):
        values = []
        for relation, pairs in enumerate(graph):
            if not pairs.shape[1]: continue
            a, b = self.schema[relation]
            # Subtract the uniform model's NLL for readable diagnostics;
            # this is a parameter-independent constant.
            log_uniform_nll = np.log(self.counts[a]*self.counts[b])
            values.append(-self.log_density(pairs, relation).mean()-log_uniform_nll)
        if not values: raise ValueError('No support observations for the density objective')
        self.circuit_diagnostics['auxiliary_calls'] += 1
        self.circuit_diagnostics['auxiliary_facts'] += sum(p.shape[1] for p in graph)
        return torch.stack(values).mean()


def setup(a, p, kind, device):
    if set(a)-TRAIN_KEYS: raise ValueError('Circuit setup accepts training arrays only')
    if kind not in KINDS: raise ValueError('Unregistered circuit arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        model = IdentityResidual().to(device)
        model.circuit_diagnostics = {}
        return model, [], [None]*p['folds'], folds
    schema, target, _ = relation_schema(a)
    graphs = []
    for fold in range(p['folds']):
        omitted = np.flatnonzero(a['folds'] == fold)
        edges, labels = graph_lib.masked_physical(a, omitted)
        relations = [torch.as_tensor(edges[:, labels == r], device=device) for r in range(len(schema)-1)]
        zero = (a['folds'] != fold) & (a['y_train'] == 0)
        zero_pairs = a['pairs_train'][:, zero]+a['pair_offset'][:, None]
        relations.append(torch.as_tensor(zero_pairs, device=device))
        graphs.append(relations)
    model = SumSquaresCircuit(a['node_counts'], schema, target, p, kind).to(device)
    return model, [], graphs, folds
