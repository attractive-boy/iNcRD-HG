"""Shift-supervised mixture of aligned experts inspired by GraphMETRO.

NeurIPS 2024, arXiv:2312.04693v3. Independent pair-level adaptation of the
paper objective. The public training script currently zeros its alignment
loss as an ablation; this implementation retains Eq. (3) with lambda=1.
"""
from datetime import datetime, timezone
import shutil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import overlap_block_experiment as common
from improve.overlap_block_models import SparseTypedAttention, masked_physical

ROOT = common.base.ROOT
OUT = ROOT / 'results/aligned_experts_core_20260915'
KINDS = ('full', 'no_alignment', 'uniform', 'erm')


def physical_tensors(edges, relations, n, nr):
    device = edges.device
    counts = torch.zeros(n * nr, device=device)
    for endpoint in edges:
        counts.index_add_(0, endpoint * nr + relations, torch.ones(len(relations), device=device))
    loops = torch.arange(n, device=device).repeat(2, 1)
    return (torch.cat((edges, edges.flip(0), loops), 1),
            torch.cat((relations, relations + nr, torch.full((n,), 2 * nr + 1, device=device, dtype=torch.long))),
            counts.reshape(n, nr).log1p())


def context(a, fold, device):
    edges, relations = masked_physical(a, np.flatnonzero(a['folds'] == fold))
    e, r = torch.as_tensor(edges, device=device), torch.as_tensor(relations, device=device)
    n, nr = int(a['node_counts'].sum()), int(a['relation'].max() + 1)
    return dict(edges=e, relations=r, n=n, nr=nr, physical=physical_tensors(e, r, n, nr))


def transformed(xs, graph, generator, code=None):
    """Four label-blind states: identity, edge removal, noise, or both."""
    device = graph['edges'].device
    code = int(torch.randint(4, (), generator=generator, device=device)) if code is None else int(code)
    if code not in range(4):
        raise ValueError('Unknown synthetic shift')
    rate = .15 + .2 * torch.rand((), generator=generator, device=device)
    keep = torch.rand(len(graph['relations']), generator=generator, device=device) >= rate
    # Selection acts on physical facts, before inverse arcs. Query facts were
    # already excluded by context(); transformations can only remove facts.
    if code & 1:
        edge, relation = graph['edges'][:, keep], graph['relations'][keep]
        physical = physical_tensors(edge, relation, graph['n'], graph['nr'])
    else:
        physical = graph['physical']
    changed = []
    for x in xs:
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
        changed.append(x + .1 * rms * noise if code & 2 else x)
    target = torch.tensor([float(code == 0), float(bool(code & 1)), float(bool(code & 2))], device=device)
    return changed, physical, target


def mix_experts(experts, gate_logits, kind):
    if kind not in KINDS:
        raise ValueError('Unknown expert control')
    if kind == 'erm':
        return experts[:, 0]
    weights = (torch.full_like(gate_logits, 1 / gate_logits.shape[-1]) if kind == 'uniform'
               else gate_logits.detach().softmax(-1))
    return (experts * weights[..., None]).sum(1)


class TypedEncoder(nn.Module):
    def __init__(self, dims, counts, relations, width, depth, dropout):
        super().__init__()
        self.counts = list(map(int, counts))
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dims])
        self.input_norm, self.output_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.layers = nn.ModuleList([SparseTypedAttention(width, 4, 2 * relations + 2, dropout)
                                     for _ in range(depth)])

    def forward(self, xs, physical):
        edges, relation, counts = physical
        values, offset = [], 0
        for x, project, size in zip(xs, self.project, self.counts):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            values.append(project(torch.cat((x / rms, counts[offset:offset + size]), 1)))
            offset += size
        hidden = F.gelu(self.input_norm(torch.cat(values)))
        for layer in self.layers:
            hidden = layer(hidden, edges, relation)
        return self.output_norm(hidden)


def pair_field(hidden, pairs):
    r, d = pairs
    return torch.cat((hidden[r], hidden[d], hidden[r] * hidden[d]), -1)


class AlignedExperts(nn.Module):
    def __init__(self, dims, node_counts, relations, kind, width=64, depth=2, dropout=.1):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unknown aligned-expert arm')
        self.kind = kind
        self.encoder = TypedEncoder(dims, node_counts, relations, width, depth, dropout)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(3 * width, 2 * width), nn.GELU(),
                                                    nn.Linear(2 * width, width)) for _ in range(3)])
        self.gate_encoder = TypedEncoder(dims, node_counts, relations, width // 2, 1, dropout)
        self.gate_head = nn.Sequential(nn.Linear(3 * (width // 2), width // 2), nn.GELU(), nn.Linear(width // 2, 3))
        self.classifier = nn.Linear(width, 1)
        self.rna_degree = nn.Linear(relations, 1)
        self.drug_degree = nn.Linear(relations, 1)

    def representations(self, xs, physical, pairs):
        field = pair_field(self.encoder(xs, physical), pairs)
        experts = torch.stack([expert(field) for expert in self.experts], 1)
        gate = self.gate_head(pair_field(self.gate_encoder(xs, physical), pairs))
        return experts, gate

    def marginal(self, physical, pairs):
        r, d = pairs
        counts = physical[2]
        return self.rna_degree(counts[r]).squeeze(-1) + self.drug_degree(counts[d]).squeeze(-1)

    def forward(self, xs, graph, pairs):
        experts, gate = self.representations(xs, graph['physical'], pairs)
        mixed = mix_experts(experts, gate, self.kind)
        return self.classifier(mixed).squeeze(-1) + self.marginal(graph['physical'], pairs)

    def training_batch(self, xs, graph, pairs, generator):
        shifted_x, shifted_graph, target = transformed(xs, graph, generator)
        used_x, used_graph = (xs, graph['physical']) if self.kind == 'erm' else (shifted_x, shifted_graph)
        experts, gate = self.representations(used_x, used_graph, pairs)
        mixed = mix_experts(experts, gate, self.kind)
        # The reference is jointly trainable, as in the paper objective and
        # author's MoE losses method. There is no EMA or stop-gradient teacher.
        clean = pair_field(self.encoder(xs, graph['physical']), pairs)
        reference = self.experts[0](clean)
        gate_loss = F.binary_cross_entropy_with_logits(gate, target[None].expand_as(gate))
        alignment = torch.linalg.vector_norm(mixed - reference) / max(len(pairs.T), 1)
        if self.kind == 'no_alignment':
            auxiliary = gate_loss
        elif self.kind == 'erm':
            auxiliary = mixed.sum() * 0.
        else:
            auxiliary = gate_loss + alignment
        score = self.classifier(mixed).squeeze(-1) + self.marginal(used_graph, pairs)
        return score, auxiliary, dict(gate=float(gate_loss.detach()), alignment=float(alignment.detach()))


def parameters():
    return dict(kinds=list(KINDS), width=64, depth=2, dropout=.1, folds=5,
                auxiliary_weight=1., primary_kind='full',
                primary_controls=['no_alignment', 'uniform', 'erm'])


def setup(a, p, kind, device):
    graphs = [context(a, fold, device) for fold in range(p['folds'])]
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == fold), device=device) for fold in range(p['folds'])]
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = AlignedExperts([x.shape[1] for x in xs], a['node_counts'], int(a['relation'].max() + 1),
                           kind, p['width'], p['depth'], p['dropout']).to(device)
    return model, xs, graphs, folds


def prepare():
    if OUT.exists():
        raise ValueError('Aligned-expert specification is already frozen')
    parent = common.protocol(common.OUT)
    sources = ('aligned_graph_experts.py', 'test_aligned_graph_experts.py',
               'overlap_block_models.py', 'overlap_block_experiment.py', 'non_tabular_experiment.py')
    evidence = ROOT / 'research/new_mechanisms_20260915'
    p = dict(**parameters(), created_at_utc=datetime.now(timezone.utc).isoformat(),
        tasks=parent['tasks'], selection=parent['selection'], specification_only=True,
        method='aligned_experts_core', family='shift_supervised_aligned_expert_mixture',
        literature=dict(title='GraphMETRO: Mitigating Complex Graph Distribution Shifts via Mixture of Aligned Experts',
                        venue='NeurIPS 2024', arxiv='2312.04693v3'),
        controls='full must beat no_alignment, uniform and erm. The same modules are initialized in all arms. no_alignment retains shifts and the gate objective; uniform retains shifts and alignment but uses uniform weights; erm trains only the reference expert on clean masked graphs. Unused modules in erm do not equal additional trained capacity.',
        adaptation='One shared 64-dimensional expert GNN with three pair-level MLP experts, separate 32-dimensional gate GNN, soft aggregation. Four equally likely transform states: identity, physical-edge removal U(.15,.35), Gaussian feature noise at .1 times per-node RMS, or both. One transform per step, not three as in the author trainer. BCE gate loss uses unweighted multi-label indicators, unlike the author code pos_weight=4. Task and alignment losses cannot backpropagate into the gate. Jointly trained clean reference expert; Frobenius norm divided by fitting-pair count; lambda=1; no warmup, no EMA.',
        author_source_difference='The retrieved author train_moe.py unconditionally sets z_dist_loss=0 under an ablation comment. This implementation follows paper Eq. (3) and retains the alignment loss; no_alignment explicitly tests its removal. No claim of running or exactly reproducing that script.',
        boundary='Delete all fitting-query target facts before relation counts and perturbations. Edge removal is applied before reverse arcs; it cannot restore queried facts. No task outcomes label the synthetic shifts. Inference uses unmodified support graphs and no query-batch statistics.',
        assumptions='Synthetic edge removal/noise is assumed to approximately preserve pair labels. This is an inductive-bias test, not evidence of natural biological distribution shifts or causal invariance.',
        input_policy='Original node attributes and masked physical relations only. No table feature expansion, pretrained models, outer outcomes or adaptive fusion weight.',
        source_sha256={n: common.base.digest(ROOT / n) for n in sources},
        inputs={str(evidence / n): common.base.digest(evidence / n)
                for n in ('graphmetro_paper.pdf', 'graphmetro_author_moe.py', 'graphmetro_author_train.py')},
        innovation_counting='One shift-aware expert-mixture family. Alignment alone is not counted separately from its trained routing and expert system.')
    OUT.mkdir(parents=True)
    (OUT / 'source_snapshot').mkdir()
    for name in p['source_sha256']:
        shutil.copy2(ROOT / name, OUT / 'source_snapshot' / name)
    common.base.write_json(OUT / 'protocol.json', p)
