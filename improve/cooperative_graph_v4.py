"""Cooperative communication with globally shared, initial-state action rates.

Full exactly reuses v3 shared_policy. The continuous control also shares its
policy across nodes; only the straight-through discrete action is removed.
This is an observed-data revision of the same cooperative mechanism family.
"""
import numpy as np
import torch

from improve import cooperative_graph_v3 as previous

KINDS = ('full', 'soft_actions')
identity_task, context = previous.identity_task, previous.context


class SharedCooperativeGraph(previous.StaticCooperativeGraph):
    def __init__(self, dimensions, relations, nodes, p, kind):
        super().__init__(dimensions, relations, nodes, p, 'shared_policy')
        self.kind = kind

    def policy_logits(self, current, initial, graph):
        logits = self.policy(self.policy_norm(initial), graph, lookahead=False)
        return logits.mean(0, keepdim=True).expand_as(logits)


def parameters():
    return previous.parameters()


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Unregistered shared cooperative arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        return previous.previous.original.IdentityResidual().to(device), [], [None] * p['folds'], folds
    graphs = [context(a, np.flatnonzero(a['folds'] == f), device) for f in range(p['folds'])]
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = SharedCooperativeGraph([x.shape[1] for x in xs], int(a['relation'].max()) + 1,
                                  int(a['node_counts'].sum()), p, kind).to(device)
    return model, xs, graphs, folds
