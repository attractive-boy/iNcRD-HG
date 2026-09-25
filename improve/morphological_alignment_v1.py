"""Functional-target-mediated drug/CRISPR morphology alignment.

This independent adaptation uses measured external phenotype profiles and
external positive DTI annotations. A mean gene-embedding proxy is not a measured
miRNA intervention, and contrastive distractors are not experimental negatives.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import tabular_foundation as base

ROOT = Path(__file__).resolve().parent
KINDS = ('full', 'no_alignment', 'shuffled_rna_targets')
ARRAYS = {'drug_profile', 'gene_profile', 'rna_gene', 'rna_gene_counts', 'rna_permutation',
          'project_drug_index', 'dti_pairs', 'drug_keys', 'gene_ids', 'drug_names', 'rna_names'}


def parameters(input_dir=None):
    folder = Path(input_dir or ROOT / 'results/morphology_inputs_v1_20260916').resolve()
    return dict(width=64, hidden=128, dropout=.1, folds=5, temperature=.1,
        alignment_weight=.1, dti_batch=256, residual_bound=2., lnc_policy='identity_strong_reference',
        input_dir=str(folder), input_manifest_sha256=base.digest(folder / 'INPUTS.json'))


def target_type(a):
    offsets = np.r_[0, np.cumsum(a['node_counts'])[:-1]]
    rna = np.flatnonzero(offsets == a['pair_offset'][0])
    drug = np.flatnonzero(offsets == a['pair_offset'][1])
    if len(rna) != 1 or len(drug) != 1 or int(rna[0]) not in (0, 1) or int(drug[0]) != 2:
        raise ValueError('Unexpected frozen RNA/drug order')
    return int(rna[0])


def identity_task(a, p):
    if p['lnc_policy'] != 'identity_strong_reference':
        raise ValueError('Undeclared LncDrug policy')
    return target_type(a) == 0


def load_inputs(p):
    folder = Path(p['input_dir'])
    if base.digest(folder / 'INPUTS.json') != p['input_manifest_sha256']:
        raise ValueError('Morphology input manifest changed')
    manifest = base.check_seal(base.read_json(folder / 'INPUTS.json'))
    if manifest['association_outcomes_read'] or manifest['new_neural_fits']:
        raise ValueError('External input provenance is not label independent')
    for name, digest in manifest['files'].items():
        if base.digest(folder / name) != digest:
            raise ValueError('Frozen morphology input changed: ' + name)
    with np.load(folder / 'arrays.npz', allow_pickle=False) as archive:
        if set(archive.files) != ARRAYS:
            raise ValueError('Unexpected morphology fields')
        a = {key: archive[key].copy() for key in archive.files}
    validate_inputs(a)
    return a, manifest


def validate_inputs(a):
    drug, gene, matrix, dti = [a[key] for key in ('drug_profile', 'gene_profile', 'rna_gene', 'dti_pairs')]
    if any(x.ndim != 2 or not np.isfinite(x).all() for x in (drug, gene, matrix)):
        raise ValueError('Invalid profile matrices')
    n_rna, n_gene = matrix.shape
    n_drug = len(drug)
    if len(gene) != n_gene or len(a['rna_names']) != n_rna or len(a['gene_ids']) != n_gene or len(a['drug_keys']) != n_drug:
        raise ValueError('Morphology dimensions do not match identities')
    counts = a['rna_gene_counts']
    actual = np.count_nonzero(matrix, axis=1)
    if counts.shape != (n_rna,) or not np.array_equal(counts, actual) or np.any(matrix < 0):
        raise ValueError('Invalid functional target-set counts')
    np.testing.assert_allclose(matrix.sum(1), (counts > 0).astype(float), atol=2e-6, rtol=0)
    for i in range(n_rna):
        if counts[i] and not np.allclose(matrix[i, matrix[i] > 0], 1. / counts[i], atol=1e-8, rtol=1e-6):
            raise ValueError('Nonuniform functional target-set weights')
    permutation = a['rna_permutation']
    if permutation.shape != (n_gene,) or not np.array_equal(np.sort(permutation), np.arange(n_gene)):
        raise ValueError('Not a target-identity permutation')
    project = a['project_drug_index']
    if project.ndim != 1 or len(project) != len(a['drug_names']) or np.any(project < -1) or np.any(project >= n_drug):
        raise ValueError('Invalid project-to-profile dictionary')
    if dti.ndim != 2 or dti.shape[0] != 2 or dti.shape[1] < 1 or not np.issubdtype(dti.dtype, np.integer):
        raise ValueError('Invalid external positive pair table')
    if np.any(dti < 0) or np.any(dti[0] >= n_drug) or np.any(dti[1] >= n_gene):
        raise ValueError('External DTI index outside phenotype dictionary')
    if len(np.unique(dti[0] * n_gene + dti[1])) != dti.shape[1]:
        raise ValueError('Duplicate external positive pair')


class IdentityResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()

    def forward(self, xs, graph, pairs):
        return torch.zeros(pairs.shape[1], dtype=torch.float32, device=pairs.device)


class Projection(nn.Module):
    def __init__(self, dimension, p):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, p['hidden']), nn.LayerNorm(p['hidden']),
            nn.ReLU(), nn.Dropout(p['dropout']), nn.Linear(p['hidden'], p['width']),
            nn.LayerNorm(p['width']), nn.ReLU(), nn.Dropout(p['dropout']), nn.Linear(p['width'], p['width']))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class MorphologicalAlignment(nn.Module):
    def __init__(self, drug_dimension, gene_dimension, offset, p, kind):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unknown morphological arm')
        self.kind = kind
        self.bound = p['residual_bound']
        self.drug = Projection(drug_dimension, p)
        self.gene = Projection(gene_dimension, p)
        self.readout = nn.Sequential(nn.Linear(4 * p['width'], p['width']), nn.GELU(),
                                    nn.Dropout(p['dropout']), nn.Linear(p['width'], 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.register_buffer('pair_offset', torch.as_tensor(offset, dtype=torch.long))
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()

    def encode(self, xs):
        drug = self.drug(xs['drug_profile'])
        gene = self.gene(xs['gene_profile'])
        matrix = xs['rna_gene'] if self.kind != 'shuffled_rna_targets' else xs['rna_gene_shuffled']
        rna = F.normalize(matrix @ gene, dim=-1)
        return drug, gene, rna

    def covered(self, xs, pairs):
        local = pairs - self.pair_offset[:, None]
        r, d = local[0], local[1]
        if torch.any(r < 0) or torch.any(r >= len(xs['rna_available'])) or torch.any(d < 0) or torch.any(d >= len(xs['project_drug_index'])):
            raise ValueError('Query outside the frozen phenotype dictionaries')
        return xs['rna_available'][r] & (xs['project_drug_index'][d] >= 0)

    def residual(self, encoded, xs, pairs):
        drug, gene, rna = encoded
        local = pairs - self.pair_offset[:, None]
        covered = self.covered(xs, pairs)
        left = rna[local[0]]
        right = drug[xs['project_drug_index'][local[1]].clamp_min(0)]
        features = torch.cat((left, right, left * right, (left - right).abs()), dim=-1)
        value = self.bound * torch.tanh(self.readout(features).squeeze(-1))
        return torch.where(covered, value, torch.zeros_like(value))

    def forward(self, xs, graph, pairs):
        if graph is not None:
            raise ValueError('This mechanism must not receive an association graph')
        return self.residual(self.encode(xs), xs, pairs)


def positive_mask(keys, drug, gene, n_gene):
    query = drug[:, None] * n_gene + gene[None, :]
    positions = torch.searchsorted(keys, query)
    return (positions < len(keys)) & (keys[positions.clamp_max(len(keys) - 1)] == query)


def multi_positive_loss(logits, mask):
    if logits.shape != mask.shape or mask.dtype != torch.bool or not mask.any(1).all() or not mask.any(0).all():
        raise ValueError('Every sampled external entity needs an observed positive')
    def directional(value, positives):
        log_prob = F.log_softmax(value, dim=1)
        return -((log_prob * positives).sum(1) / positives.sum(1)).mean()
    return (directional(logits, mask) + directional(logits.T, mask.T)) / 2


def alignment_loss(encoded, xs, p, generator):
    drug, gene, rna = encoded
    selection = torch.randint(xs['dti_pairs'].shape[1], (p['dti_batch'],), generator=generator, device=drug.device)
    pairs = xs['dti_pairs'][:, selection]
    mask = positive_mask(xs['dti_keys'], pairs[0], pairs[1], len(gene))
    if not torch.diagonal(mask).all():
        raise ValueError('Sampled DTI is not a registered positive')
    logits = drug[pairs[0]] @ gene[pairs[1]].T / p['temperature']
    loss = multi_positive_loss(logits, mask)
    return loss, dict(sampled_pairs=len(selection), positive_mask_entries=int(mask.sum()),
        unique_drug_entities=int(pairs[0].unique().numel()), unique_gene_entities=int(pairs[1].unique().numel()),
        external_positive_supervision_verified=True)


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Unregistered morphology arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        return IdentityResidual().to(device), {}, [None] * p['folds'], folds
    z, manifest = load_inputs(p)
    if int(a['node_counts'][1]) != len(z['rna_names']) or int(a['node_counts'][2]) != len(z['drug_names']):
        raise ValueError('External entities do not match frozen project dictionaries')
    xs = {key: torch.as_tensor(z[key], device=device) for key in
          ('drug_profile', 'gene_profile', 'rna_gene', 'project_drug_index', 'dti_pairs')}
    xs['rna_gene_shuffled'] = torch.as_tensor(z['rna_gene'][:, z['rna_permutation']], device=device)
    xs['rna_available'] = torch.as_tensor(z['rna_gene_counts'] > 0, device=device)
    xs['dti_keys'] = torch.sort(xs['dti_pairs'][0] * len(xs['gene_profile']) + xs['dti_pairs'][1]).values
    model = MorphologicalAlignment(z['drug_profile'].shape[1], z['gene_profile'].shape[1], a['pair_offset'], p, kind).to(device)
    # All five training folds see only the same frozen external inputs. No
    # association edge, attribute or outcome is used to construct the encoder.
    return model, xs, [None] * p['folds'], folds
