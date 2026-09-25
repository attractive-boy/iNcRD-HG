"""Typed complex Stuart-Landau propagation with an implicit radial update.

The local-then-old-state-coupling order follows the pinned author's ode_step,
not the different ordering in the paper's Algorithm 1. The positive cubic
root is evaluated in squared radius, with its implicit derivative, so the
Cartesian map has its correct smooth extension at zero amplitude.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import nested_anchor_inputs_v2 as nested
from improve import overlap_block_models as graph_lib
from improve.sharpness_graph_v1 import IdentityResidual, identity_task

KINDS = ('full', 'no_phase_feedback', 'linear_oscillator')
TRAIN_KEYS = nested.TRAIN_KEYS | {'anchor_oof'}


def parameters():
    return dict(width=32, depth=16, readout_width=64, dropout=.1, folds=5,
        dt=.1, alpha=1., beta=1., omega=1., gamma=.5, coupling_strength=1.,
        leaky_relu_slope=.2, local_dtype='float64', coupling_dtype='float32',
        radial_solver='unique positive cubic via hyperbolic inverse in squared radius',
        radial_gradient='implicit derivative including zero-radius extension',
        radial_residual_tolerance=2e-12, local_order='local update plus explicit coupling of old state',
        coupling='symmetric-degree-normalized complex linear GCN with typed positive channel gates',
        coupling_imaginary_weight_initialization='zero; real weight Xavier-uniform',
        graph_self_loops='preserve the common physical graph convention',
        node_readout='real component, following source SLGCN',
        lnc_policy='identity_strong_reference')


def empty_diagnostics():
    return dict(calls=0, local_steps=0, radial_solves=0, solver_entries=0,
        coupling_messages=0, maximum_radial_relative_residual=0., maximum_state_amplitude=0.)


class SquaredRadialRoot(torch.autograd.Function):
    """q >= 0 solves q*(a+s*q)^2=v for a>0, s>0, v>=0.

This is equivalent to s*R^3+a*R=sqrt(v), with q=R^2. The hyperbolic
inverse avoids subtraction of nearly equal Cardano terms. Differentiation
uses the monotone implicit equation, including its regular v=0 boundary.
"""
    @staticmethod
    def forward(ctx, v, a, s):
        if (v.dtype != torch.float64 or a.dtype != v.dtype or s.dtype != v.dtype
                or not torch.isfinite(v).all() or not torch.isfinite(a).all()
                or not torch.isfinite(s).all() or torch.any(v < 0)
                or torch.any(a <= 0) or torch.any(s <= 0)):
            raise ValueError('Squared radial solve requires finite FP64 v>=0, a>0 and s>0')
        p = a / (3.*s)
        argument = torch.sqrt(v) / (2.*s*p.pow(1.5))
        radius = 2.*torch.sqrt(p)*torch.sinh(torch.asinh(argument)/3.)
        q = radius.square()
        if not torch.isfinite(q).all():
            raise FloatingPointError('Nonfinite squared radial root')
        ctx.save_for_backward(q, a, s)
        ctx.v_shape, ctx.a_shape, ctx.s_shape = v.shape, a.shape, s.shape
        return q

    @staticmethod
    def backward(ctx, upstream):
        q, a, s = ctx.saved_tensors
        first, third = a+s*q, a+3.*s*q
        dv = upstream / (first*third)
        da = upstream * (-2.*q/third)
        ds = upstream * (-2.*q.square()/third)
        return dv.sum_to_size(ctx.v_shape), da.sum_to_size(ctx.a_shape), ds.sum_to_size(ctx.s_shape)


def local_step(state, p, kind):
    if state.shape[-1] != 2 or kind not in KINDS:
        raise ValueError('Expected Cartesian oscillator states and a registered arm')
    original_dtype = state.dtype
    state64 = state.double()
    real, imaginary = state64.unbind(-1)
    squared = real.square() + imaginary.square()
    a = state64.new_tensor(1.-p['dt']*p['alpha'])
    s = state64.new_tensor(p['dt']*p['beta'])
    q = SquaredRadialRoot.apply(squared, a, s)
    residual = (q*(a+s*q).square()-squared).abs() / (1.+squared)
    if torch.any(residual > p['radial_residual_tolerance']):
        raise FloatingPointError('Registered implicit radial residual tolerance was exceeded')
    scale = 1./(a+s*q)
    angle = p['dt']*(p['omega']-p['gamma']*q)
    fixed_angle = p['dt']*p['omega']
    def rotate(cosine, sine, radial_scale):
        return torch.stack((radial_scale*(real*cosine-imaginary*sine),
                            radial_scale*(real*sine+imaginary*cosine)), dim=-1)
    full = rotate(torch.cos(angle), torch.sin(angle), scale)
    cosine, sine = np.cos(fixed_angle), np.sin(fixed_angle)
    no_feedback = rotate(cosine, sine, scale)
    linear = rotate(cosine, sine, 1./a)
    # Every arm computes the same nonlinear radial solver. Multiplying an
    # unused local path by zero is budget accounting, not a capacity claim.
    if kind == 'full':
        output = full + 0.*(no_feedback+linear)
    elif kind == 'no_phase_feedback':
        output = no_feedback + 0.*(full+linear)
    else:
        output = linear + 0.*(full+no_feedback)
    if not torch.isfinite(output).all():
        raise FloatingPointError('Nonfinite Stuart-Landau local update')
    return output.to(original_dtype), dict(radial_solves=1, solver_entries=squared.numel(),
        maximum_radial_relative_residual=float(residual.detach().max()) if residual.numel() else 0.)


@dataclass
class OscillatorContext:
    physical: tuple
    degree: torch.Tensor
    normalization: torch.Tensor


def pack_graph(physical):
    edges, _, counts = physical
    degree = torch.bincount(edges[1], minlength=len(counts)).to(torch.float32)
    inverse = degree.clamp_min(1.).rsqrt()
    return OscillatorContext(physical, degree, inverse[edges[0]]*inverse[edges[1]])


class ComplexTypedGCN(nn.Module):
    def __init__(self, width, relations, slope):
        super().__init__()
        self.real_weight = nn.Parameter(torch.empty(width, width))
        self.imaginary_weight = nn.Parameter(torch.zeros(width, width))
        self.relation_gate = nn.Parameter(torch.zeros(relations, width))
        nn.init.xavier_uniform_(self.real_weight)
        self.slope = slope

    def forward(self, state, context):
        edges, relation, _ = context.physical
        real, imaginary = state.unbind(-1)
        # A complex-linear channel map; imaginary weights start at zero,
        # matching the author's real GCN weight converted to complex dtype.
        u = F.linear(real, self.real_weight) - F.linear(imaginary, self.imaginary_weight)
        v = F.linear(real, self.imaginary_weight) + F.linear(imaginary, self.real_weight)
        gate = 2.*torch.sigmoid(self.relation_gate[relation])
        weight = context.normalization[:, None]*gate
        output_u = torch.zeros_like(u).index_add(0, edges[1], u[edges[0]]*weight)
        output_v = torch.zeros_like(v).index_add(0, edges[1], v[edges[0]]*weight)
        return torch.stack((F.leaky_relu(output_u, self.slope),
                            F.leaky_relu(output_v, self.slope)), dim=-1)


class StuartLandauGraph(nn.Module):
    def __init__(self, dimensions, relations, p, kind):
        super().__init__()
        if kind not in KINDS or p['beta'] <= 0 or not 1.-p['dt']*p['alpha'] > 0:
            raise ValueError('Unregistered arm or nonunique radial solve regime')
        self.p, self.kind = dict(p), kind
        width = p['width']
        self.project = nn.ModuleList([nn.Linear(d+relations, 2*width) for d in dimensions])
        self.couplings = nn.ModuleList([ComplexTypedGCN(width, 2*relations+2,
            p['leaky_relu_slope']) for _ in range(p['depth'])])
        self.readout = nn.Sequential(nn.Linear(4*width, p['readout_width']), nn.GELU(),
            nn.Dropout(p['dropout']), nn.Linear(p['readout_width'], 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()
        self.oscillator_diagnostics = empty_diagnostics()

    def encode(self, xs, context):
        counts = context.physical[2]
        parts, start = [], 0
        for x, projection in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            inputs = torch.cat((x/rms, counts[start:start+len(x)]), dim=1)
            inputs = F.dropout(inputs, p=self.p['dropout'], training=self.training)
            parts.append(projection(inputs).reshape(len(x), self.p['width'], 2))
            start += len(x)
        state = torch.cat(parts)
        d = self.oscillator_diagnostics
        d['calls'] += 1
        for coupling in self.couplings:
            updated, audit = local_step(state, self.p, self.kind)
            # This follows pinned SLGCN.ode_step: local(old) + dt*k*F(old).
            state = updated + self.p['dt']*self.p['coupling_strength']*coupling(state, context)
            d['local_steps'] += 1
            d['radial_solves'] += audit['radial_solves']
            d['solver_entries'] += audit['solver_entries']
            d['coupling_messages'] += context.physical[0].shape[1]
            d['maximum_radial_relative_residual'] = max(d['maximum_radial_relative_residual'],
                                                        audit['maximum_radial_relative_residual'])
            with torch.no_grad():
                if not torch.isfinite(state).all():
                    raise FloatingPointError('Nonfinite coupled Stuart-Landau state')
                d['maximum_state_amplitude'] = max(d['maximum_state_amplitude'],
                    float(torch.linalg.vector_norm(state, dim=-1).max()))
        # Source SLGCN decodes the real component. No phase/amplitude values
        # are appended to a tabular feature bank or normalized away here.
        return F.dropout(state[..., 0], p=self.p['dropout'], training=self.training)

    def forward(self, xs, context, pairs):
        hidden = self.encode(xs, context)
        left, right = hidden[pairs[0]], hidden[pairs[1]]
        return self.readout(torch.cat((left, right, left*right, (left-right).abs()), dim=1)).squeeze(-1)


def setup(a, p, kind, device):
    if set(a) - TRAIN_KEYS:
        raise ValueError('Stuart-Landau setup accepts training arrays only')
    if kind not in KINDS:
        raise ValueError('Unregistered oscillator arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        return IdentityResidual().to(device), [], [None]*p['folds'], folds
    n, relations = int(a['node_counts'].sum()), int(a['relation'].max())+1
    graphs = []
    for f in range(p['folds']):
        edges, labels = graph_lib.masked_physical(a, np.flatnonzero(a['folds'] == f))
        graphs.append(pack_graph(graph_lib.graph_inputs(edges, labels,
            np.empty((2, 0), dtype=np.int64), n, relations, device)))
    xs = [torch.as_tensor(a['x_'+str(i)], device=device) for i in range(len(a['node_counts']))]
    model = StuartLandauGraph([x.shape[1] for x in xs], relations, p, kind).to(device)
    return model, xs, graphs, folds
