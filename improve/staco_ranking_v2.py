"""Observed-data ranking revision with both proximal terms identically disabled.

The primary exactly replays v1 no_proximal. The new BCE control has the same
zero penalty, private blocks and stage averaging. This is not STACO2 and does
not claim a nonconvex convergence guarantee from the STACO1 analysis.
"""
from improve import staco_ranking_v1 as previous

KINDS = ('full', 'bce')
identity_task, parameters = previous.identity_task, previous.parameters


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Unregistered nonproximal ranking arm')
    return previous.setup(a, p, 'no_proximal', device)


class TailState(previous.TailState):
    def block_objective(self, probability, batch, kind):
        if kind not in KINDS:
            raise ValueError('Unregistered nonproximal ranking arm')
        return super().block_objective(probability, batch, 'no_proximal')

    def parameter_penalty(self, model, kind):
        if kind not in KINDS:
            raise ValueError('Unregistered nonproximal ranking arm')
        return super().parameter_penalty(model, 'no_proximal')
