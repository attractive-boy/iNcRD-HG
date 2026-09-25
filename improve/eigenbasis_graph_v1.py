"""Exact parent architecture with new, explicitly declared optimizer labels."""
from improve import preconditioned_graph_v1 as parent
from improve.eigenbasis_adam_v1 import KINDS

identity_task, parameters = parent.identity_task, parent.parameters


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Undeclared eigenbasis optimizer')
    return parent.setup(a, p, 'soap' if kind != 'adamw' else 'adamw', device)
