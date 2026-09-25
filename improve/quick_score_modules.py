"""Small, query-independent adapters for already frozen association scores.

These rules use no evaluation labels. Entity rates are rates of observed
database labels in the supplied support, not biological response prevalence.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import expit


EPSILON = 1e-12
PRIOR_MASS = 20.0
CORRECTION = 0.25
CANDIDATES = (
    'logit_pool', 'coverage_gate', 'rna_frequency_correction',
    'drug_frequency_correction',
)
CONTROLS = ('coverage_gate_shuffled',)
VARIANTS = ('reference', *CANDIDATES, *CONTROLS)


def log_odds(probability):
    p = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Expected finite probabilities between zero and one')
    p = np.clip(p, EPSILON, 1.0 - EPSILON)
    return np.log(p) - np.log1p(-p)


def checked_pairs(pairs, sizes):
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[0] != 2 or pairs.dtype.kind not in 'iu':
        raise ValueError('Expected a (2, n) integer entity-pair array')
    if (pairs < 0).any() or any((pairs[i] >= size).any() for i, size in enumerate(sizes)):
        raise ValueError('Entity ID outside the frozen dictionary')
    return pairs


def observed_offsets(pairs, labels, sizes):
    pairs = checked_pairs(pairs, sizes)
    labels = np.asarray(labels)
    if labels.shape != (pairs.shape[1],) or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Support requires aligned binary observed labels')
    if len(set(map(tuple, pairs.T))) != len(labels):
        raise ValueError('Duplicate support pairs')
    prevalence = float(labels.mean())
    offsets, positive_counts = [], []
    for axis, size in enumerate(sizes):
        positive = np.bincount(pairs[axis], weights=labels, minlength=size)
        total = np.bincount(pairs[axis], minlength=size)
        rate = (positive + PRIOR_MASS * prevalence) / (total + PRIOR_MASS)
        offsets.append(log_odds(rate) - log_odds(prevalence))
        positive_counts.append(positive)
    return offsets, positive_counts, prevalence


@dataclass
class FrozenScoreAdapter:
    rna_offset: np.ndarray
    drug_offset: np.ndarray
    rna_evidence: np.ndarray
    drug_evidence: np.ndarray
    shuffled_rna_evidence: np.ndarray
    shuffled_drug_evidence: np.ndarray

    @classmethod
    def fit(cls, pairs, labels, rna_evidence, drug_evidence, seed=20260914):
        rna_evidence = np.asarray(rna_evidence, dtype=np.float64)
        drug_evidence = np.asarray(drug_evidence, dtype=np.float64)
        for vector in (rna_evidence, drug_evidence):
            if vector.ndim != 1 or not np.isfinite(vector).all() or (vector < 0).any() or (vector > 1).any():
                raise ValueError('Evidence strength must be a finite vector in [0, 1]')
        offsets, _, _ = observed_offsets(pairs, labels, (len(rna_evidence), len(drug_evidence)))
        rng = np.random.default_rng(seed)
        return cls(*offsets, rna_evidence.copy(), drug_evidence.copy(),
                   rng.permutation(rna_evidence), rng.permutation(drug_evidence))

    def confidence(self, pairs, shuffled=False):
        pairs = checked_pairs(pairs, (len(self.rna_offset), len(self.drug_offset)))
        rna = self.shuffled_rna_evidence if shuffled else self.rna_evidence
        drug = self.shuffled_drug_evidence if shuffled else self.drug_evidence
        return rna[pairs[0]] * drug[pairs[1]]

    def predict(self, graph, tiered, pairs, variant):
        if variant not in VARIANTS:
            raise ValueError('Unregistered score adapter')
        pairs = checked_pairs(pairs, (len(self.rna_offset), len(self.drug_offset)))
        graph, tiered = np.asarray(graph, dtype=np.float64), np.asarray(tiered, dtype=np.float64)
        if graph.shape != (pairs.shape[1],) or tiered.shape != graph.shape:
            raise ValueError('Scores do not align with entity pairs')
        # Validate even for the arithmetic branch.
        graph_logit, tiered_logit = log_odds(graph), log_odds(tiered)
        reference = 0.5 * graph + 0.5 * tiered
        if variant == 'reference':
            return reference
        if variant == 'logit_pool':
            return expit(0.5 * graph_logit + 0.5 * tiered_logit)
        if variant.startswith('coverage_gate'):
            alpha = 0.25 + 0.5 * self.confidence(pairs, shuffled=variant.endswith('shuffled'))
            return (1.0 - alpha) * graph + alpha * tiered
        offset = self.rna_offset[pairs[0]] if variant.startswith('rna_') else self.drug_offset[pairs[1]]
        return expit(log_odds(reference) - CORRECTION * offset)

    def save(self, path):
        np.savez_compressed(path, **vars(self))

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as archive:
            if set(archive.files) != set(cls.__dataclass_fields__):
                raise ValueError('Unexpected adapter state')
            return cls(**{name: archive[name].copy() for name in archive.files})
