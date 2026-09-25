"""ConFIG-inspired gradient integration for the two RNA association tasks.

All arms use the same joint support graph. The other RNA task contributes
training observations only; neither task's evaluation facts enter the graph.
The primary task's original attribute arrays define the shared input basis.
ConFIG's guarantee concerns the proposed raw gradient, before AdamW.
"""
from datetime import datetime, timezone
import math
import shutil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import overlap_block_experiment as common
from improve.overlap_block_models import SparseTypedAttention, graph_inputs

ROOT = common.base.ROOT
OUT = ROOT / 'results/conflict_free_core_20260915'
KINDS = ('full', 'sum', 'pcgrad', 'main_only')
SECONDARY_KEYS = frozenset(('pairs_train', 'y_train', 'folds', 'edges', 'relation',
    'target_codes', 'node_counts', 'nd', 'pair_offset', 'offset_nodes', 'offset_intercepts'))
EXTRA_KEYS = frozenset(('secondary_pairs_train', 'secondary_y_train', 'secondary_folds',
    'secondary_pair_offset', 'secondary_offset_nodes', 'secondary_offset_intercepts',
    'secondary_target_codes'))


def target_relation(a):
    mask = a['target_codes'] >= 0
    relations = np.unique(a['relation'][mask])
    if len(relations) != 1:
        raise ValueError('Expected one nonempty target relation')
    expected = a['pairs_train'][:, a['y_train'] == 1]
    expected_codes = expected[0] * int(a['nd']) + expected[1]
    np.testing.assert_array_equal(np.sort(a['target_codes'][mask]), np.sort(expected_codes))
    return int(relations[0])


def add_secondary_training(main, secondary):
    """Augment a new array mapping; never mutate either sealed source mapping."""
    if set(secondary) != SECONDARY_KEYS:
        raise ValueError('Secondary task accepts training inputs only, with all required fields')
    if EXTRA_KEYS.intersection(main):
        raise ValueError('Secondary observations are already present')
    np.testing.assert_array_equal(main['node_counts'], secondary['node_counts'])
    if int(main['nd']) != int(secondary['nd']) or main['pair_offset'][1] != secondary['pair_offset'][1]:
        raise ValueError('Drug index space mismatch')
    if main['pair_offset'][0] == secondary['pair_offset'][0]:
        raise ValueError('The auxiliary task must use the other RNA type')
    primary_rel, secondary_rel = target_relation(main), target_relation(secondary)
    if primary_rel == secondary_rel:
        raise ValueError('Target relation IDs must differ')
    # Discard the historical other-task fact set. Insert exactly its fitting
    # positives, so its evaluation outcomes cannot enter the shared encoder.
    keep = main['relation'] != secondary_rel
    positives = secondary['pairs_train'][:, secondary['y_train'] == 1]
    extra_edges = positives + secondary['pair_offset'][:, None]
    output = dict(main)
    output['edges'] = np.concatenate((main['edges'][:, keep], extra_edges), axis=1)
    output['relation'] = np.concatenate((main['relation'][keep], np.full(positives.shape[1], secondary_rel, dtype=np.int64)))
    output['target_codes'] = np.concatenate((main['target_codes'][keep], np.full(positives.shape[1], -1, dtype=np.int64)))
    output['secondary_target_codes'] = np.concatenate((np.full(int(keep.sum()), -1, dtype=np.int64),
                                                       positives[0] * int(main['nd']) + positives[1]))
    for key in ('pairs_train', 'y_train', 'folds', 'pair_offset', 'offset_nodes', 'offset_intercepts'):
        output['secondary_' + key] = secondary[key].copy()
    return output


def context(a, fold, device):
    primary = a['pairs_train'][:, a['folds'] == fold]
    secondary = a['secondary_pairs_train'][:, a['secondary_folds'] == fold]
    pcode = primary[0] * int(a['nd']) + primary[1]
    scode = secondary[0] * int(a['nd']) + secondary[1]
    keep = ~np.isin(a['target_codes'], pcode) & ~np.isin(a['secondary_target_codes'], scode)
    return graph_inputs(a['edges'][:, keep], a['relation'][keep], np.empty((2, 0), dtype=np.int64),
                        int(a['node_counts'].sum()), int(a['relation'].max() + 1), device)


@torch.no_grad()
def combine_gradients(gradients, kind):
    """Equal-direction-weight ConFIG, exact for two nonzero task gradients.

    If u_i=g_i/||g_i||, equations 2/3 reduce to
    (||g_1||+||g_2||)(u_1+u_2)/2. Opposing directions give zero, since no
    strict common descent exists. One zero gradient leaves the other intact.
    Dot products and norms use float64 before restoring parameter precision.
    """
    if kind not in KINDS or gradients.ndim != 2 or gradients.shape[0] != 2:
        raise ValueError('Expected two task gradients and a registered rule')
    if not torch.isfinite(gradients).all():
        raise FloatingPointError('Nonfinite task gradient')
    g = gradients.double()
    if kind == 'main_only':
        return gradients[0].clone()
    if kind == 'sum':
        return g.sum(0).to(gradients.dtype)
    norms = torch.linalg.vector_norm(g, dim=1)
    if (norms == 0).any():
        return g.sum(0).to(gradients.dtype)
    if kind == 'full':
        result = .5 * norms.sum() * (g / norms[:, None]).sum(0)
    else:
        dot = g[0] @ g[1]
        correction = torch.minimum(dot, dot.new_zeros(()))
        result = (g[0] - correction / norms[1].square() * g[1]
                  + g[1] - correction / norms[0].square() * g[0])
    return result.to(gradients.dtype)


def gradient_vector(loss, parameters, retain_graph=False):
    pieces = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p, g in zip(parameters, pieces)]).detach()


def apply_gradient(parameters, gradient):
    offset = 0
    for parameter in parameters:
        parameter.grad = gradient[offset:offset + parameter.numel()].reshape_as(parameter)
        offset += parameter.numel()
    if offset != gradient.numel():
        raise ValueError('Parameter/gradient size mismatch')


@torch.no_grad()
def cosine_rows(rows, vector):
    x, y = rows.double(), vector.double()
    denominator = torch.linalg.vector_norm(x, dim=1) * torch.linalg.vector_norm(y)
    return (x @ y) / denominator.clamp_min(torch.finfo(torch.float64).tiny)


class JointGraph(nn.Module):
    """A conventional shared typed encoder, identical in every optimizer arm."""
    def __init__(self, dims, node_counts, relations, width=64, depth=2, heads=4, dropout=.1):
        super().__init__()
        self.node_counts = list(map(int, node_counts))
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dims])
        self.input_norm, self.output_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.layers = nn.ModuleList([SparseTypedAttention(width, heads, 2 * relations + 2, dropout)
                                     for _ in range(depth)])
        self.rna_readout = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.drug_readout = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.rna_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))
        self.drug_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))
        self.scale = math.sqrt(width)

    def forward(self, xs, graph, pairs):
        edges, relations, counts = graph
        pieces, offset = [], 0
        for x, project, size in zip(xs, self.project, self.node_counts):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            pieces.append(project(torch.cat((x / rms, counts[offset:offset + size]), 1)))
            offset += size
        hidden = F.gelu(self.input_norm(torch.cat(pieces)))
        for layer in self.layers:
            hidden = layer(hidden, edges, relations)
        hidden = self.output_norm(hidden)
        r, d = pairs
        score = (self.rna_readout(hidden)[r] * self.drug_readout(hidden)[d]).sum(-1) / self.scale
        return score + self.rna_degree(counts[r]).squeeze(-1) + self.drug_degree(counts[d]).squeeze(-1)


def parameters():
    return dict(kinds=list(KINDS), width=64, depth=2, heads=4, dropout=.1, folds=5,
                auxiliary_weight=0., primary_kind='full', primary_controls=['sum', 'pcgrad', 'main_only'])


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Unknown optimizer arm')
    graphs = [context(a, fold, device) for fold in range(p['folds'])]
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == fold), device=device) for fold in range(p['folds'])]
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = JointGraph([x.shape[1] for x in xs], a['node_counts'], int(a['relation'].max() + 1),
                      p['width'], p['depth'], p['heads'], p['dropout']).to(device)
    return model, xs, graphs, folds


def prepare():
    if OUT.exists():
        raise ValueError('Use a new conflict-free training specification directory')
    parent = common.protocol(common.OUT)
    sources = ('conflict_free_graph_training.py', 'test_conflict_free_graph_training.py',
               'overlap_block_models.py', 'overlap_block_experiment.py', 'non_tabular_experiment.py')
    files = [ROOT / 'research/new_mechanisms_20260915' / n for n in
             ('config_paper.pdf', 'config_author_grad_operator.py', 'config_author_length_model.py')]
    p = dict(**parameters(), created_at_utc=datetime.now(timezone.utc).isoformat(),
        tasks=parent['tasks'], selection=parent['selection'], specification_only=True,
        method='conflict_free_core', family='conflict_free_multitask_optimization',
        literature=dict(title='ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks',
                        venue='ICLR 2025', arxiv='2408.11104v3'),
        controls='Identical shared graph encoder, raw parameters, fitting graph, offsets and budget. The two task BCE gradients are independently evaluated for every arm. full uses equal-direction ConFIG with projection length, sum adds the gradients, pcgrad projects conflicting task gradients, main_only uses only the primary gradient. Full must beat all three on mean MiDrug AUC and AP. Shared multi-task architecture itself is not the innovation.',
        boundary='Replace the other-task relation with exactly its paired training positives. Mask the current fitting fold of BOTH tasks before counts and reverse arcs; no other-task validation pairs or outcomes enter training construction. Both task offsets reuse the already sealed support-fold fits. Main task original attributes, including its fixed PCA basis, are kept for both losses; secondary attribute arrays are not consumed.',
        adaptation='Independent application to RNA association BCE losses, not PINN reproduction. A shared 64-wide two-layer typed attention encoder is evaluated once for both task pair lists per step, so both gradients use the same dropout realization. ConFIG combines all trainable parameters and precedes the common AdamW and norm clipping. No momentum-accelerated M-ConFIG, no additional loss weights, no prior-model replacement.',
        theory_scope='ConFIG raw gradients have nonnegative products with both task gradients, except numerical rounding. Exact opposing gradients produce zero, and a zero task gradient leaves the other unchanged. This is not a guarantee that the AdamW step, finite stochastic losses, ranking metrics or both validation tasks improve. Raw and actual-step conflicts are recorded separately.',
        source_difference='The local algebraic two-gradient expression is checked against the normalized-gradient pseudoinverse definition. The paper main text says parameter count alone assures existence; Appendix A.3 requires the rank condition. The main text claim that PCGrad and ConFIG share a direction for any two losses does not hold for unequal gradient norms; controls implement the actual separate definitions.',
        source_sha256={n: common.base.digest(ROOT / n) for n in sources},
        inputs={str(p): common.base.digest(p) for p in files},
        innovation_counting='One multi-objective gradient-integration family. Neither shared architecture, classical offsets, fusion nor the separate RNA-task models are additional innovations.')
    OUT.mkdir(parents=True)
    (OUT / 'source_snapshot').mkdir()
    for name in p['source_sha256']:
        shutil.copy2(ROOT / name, OUT / 'source_snapshot' / name)
    common.base.write_json(OUT / 'protocol.json', p)
