"""Same cooperative family, testing local-state policies without look-ahead.

Full matches the previous no_lookahead arm. New controls isolate the discrete
action and dynamic-state mechanisms; previous failed conclusions are unchanged.
"""
import numpy as np
import torch

from improve import cooperative_graph_v1 as original

KINDS = ('full', 'soft_actions', 'static_policy')
identity_task, context = original.identity_task, original.context


class LocalCooperativeGraph(original.CooperativeGraph):
    def __init__(self, dimensions, relations, nodes, p, kind):
        super().__init__(dimensions, relations, nodes, p, 'no_lookahead')
        self.kind = kind

    def policy_logits(self, current, initial, graph):
        state = initial if self.kind == 'static_policy' else current
        return self.policy(self.policy_norm(state), graph, lookahead=False)

    def one_draw(self, initial, graph, pairs, sample=None):
        h = initial
        for i, layer in enumerate(self.layers):
            logits = self.policy_logits(h, initial, graph)
            if sample is None:
                uniform = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
                noise = -torch.log(-torch.log(uniform))
            else:
                noise = self.evaluation_noise[sample, i]
            decisions = original.actions(logits, noise, self.temperature, hard=self.kind != 'soft_actions')
            h = layer(h, graph, original.edge_weights(decisions, graph.edges))
        left, right = h[pairs[0]], h[pairs[1]]
        return self.readout(torch.cat((left, right, left * right, (left - right).abs()), dim=-1)).squeeze(-1)


def parameters():
    return original.parameters()


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Unregistered local cooperative arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        return original.IdentityResidual().to(device), [], [None] * p['folds'], folds
    graphs = [context(a, np.flatnonzero(a['folds'] == f), device) for f in range(p['folds'])]
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = LocalCooperativeGraph([x.shape[1] for x in xs], int(a['relation'].max()) + 1,
                                  int(a['node_counts'].sum()), p, kind).to(device)
    return model, xs, graphs, folds
