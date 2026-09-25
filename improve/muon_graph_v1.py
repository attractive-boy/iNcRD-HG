"""Identical ordinary graph architecture for all spectral optimizer arms."""
from improve import sharpness_graph_v1 as ordinary

KINDS = ('full', 'frobenius', 'adamw')
identity_task = ordinary.identity_task


def parameters():
    return ordinary.parameters()


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Unregistered spectral optimizer arm')
    return ordinary.setup(a, p, 'erm', device)
