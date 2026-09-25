"""Same cooperative family: discrete communication from initial node states.

Full is exactly v2's already observed static_policy. A new soft-action control
uses the same initial-state policy; evolving-state adaptation is not claimed.
"""
import numpy as np
import torch

from improve import cooperative_graph_v2 as previous

KINDS = ('full', 'soft_actions', 'shared_policy')
identity_task, context = previous.identity_task, previous.context


class StaticCooperativeGraph(previous.LocalCooperativeGraph):
    def __init__(self, dimensions, relations, nodes, p, kind):
        super().__init__(dimensions, relations, nodes, p, 'static_policy')
        self.kind = kind

    def policy_logits(self, current, initial, graph):
        logits = self.policy(self.policy_norm(initial), graph, lookahead=False)
        if self.kind == 'shared_policy':
            logits = logits.mean(0, keepdim=True).expand_as(logits)
        return logits


def parameters():
    return previous.parameters()


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Unregistered static cooperative arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        return previous.original.IdentityResidual().to(device), [], [None] * p['folds'], folds
    graphs = [context(a, np.flatnonzero(a['folds'] == f), device) for f in range(p['folds'])]
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = StaticCooperativeGraph([x.shape[1] for x in xs], int(a['relation'].max()) + 1,
                                   int(a['node_counts'].sum()), p, kind).to(device)
    return model, xs, graphs, folds
