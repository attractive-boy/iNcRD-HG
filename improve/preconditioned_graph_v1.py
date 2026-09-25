"""Unchanged graph architecture for a prospective Kronecker optimizer study."""
from improve import sharpness_graph_v1 as ordinary
from improve import nested_anchor_inputs_v2 as nested
from improve.kronecker_preconditioning_v1 import KINDS

identity_task = ordinary.identity_task
parameters = ordinary.parameters


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Undeclared Kronecker optimizer arm')
    if set(a) - (nested.TRAIN_KEYS | {'anchor_oof'}):
        raise ValueError('Graph setup accepts training arrays only')
    return ordinary.setup(a, p, 'erm', device)
