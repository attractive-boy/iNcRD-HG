"""Narrow revision testing functional target identity without DTI alignment.

The full arm is exactly the previously observed no_alignment arm. This is an
adaptive revision of the same morphology family, not an independent discovery.
"""
from improve import morphological_alignment_v1 as parent
from improve import tabular_foundation as base

KINDS = ('full', 'shuffled_rna_targets')
identity_task = parent.identity_task
load_inputs = parent.load_inputs
alignment_loss = parent.alignment_loss


def parameters(input_dir=None):
    p = parent.parameters(input_dir)
    p.update(alignment_weight=0., parent_core_sha256=base.digest(parent.__file__))
    return p


def setup(a, p, kind, device):
    if kind not in KINDS or p['alignment_weight'] != 0:
        raise ValueError('This revision requires a registered arm and zero alignment weight')
    if p['parent_core_sha256'] != base.digest(parent.__file__):
        raise ValueError('Frozen parent phenotype implementation changed')
    mapped = 'no_alignment' if kind == 'full' else 'shuffled_rna_targets'
    return parent.setup(a, p, mapped, device)
