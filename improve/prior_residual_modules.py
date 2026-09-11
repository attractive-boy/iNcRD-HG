"""Small, bounded corrections around fixed similarity weights.

Only non-self similarity messages change. Incoming total weight and each
existing self-loop weight are preserved, as in the matched AttributeConv.
"""
import torch
from torch.nn import functional as F
from torch_geometric.utils import scatter

from improve.attention_mechanisms import norm
from improve.edge_path_modules import AttributeConv, EdgePathModel, shuffle_attributes


CONTROLS = ('base', 'edge_fixed', 'edge_attributes')
NEW_ARMS = ('prior_bounded', 'prior_free', 'bounded_uniform', 'prior_shuffled')
ARMS = CONTROLS + NEW_ARMS
PRIMARY = ('prior_bounded',)


class PriorResidualConv(AttributeConv):
    def __init__(self, original, edges, attributes, n_target, mode, seed,
                 width=16, epsilon=.25):
        if mode not in NEW_ARMS:
            raise ValueError(mode)
        if not 0 < epsilon < 1:
            raise ValueError('epsilon must lie strictly between zero and one')
        # The scorer, random draws, signed attributes and core all match the
        # existing free-attribute attention. Shuffle ONLY the multiplicative
        # prior in the shuffled control, keeping scorer inputs unchanged.
        super().__init__(original, edges, attributes, n_target,
                         'edge_attributes', seed, width)
        self.mode, self.epsilon = mode, float(epsilon)
        if mode == 'prior_shuffled':
            strength = shuffle_attributes(attributes, edges, seed)[self.nonself, 1]
        elif mode == 'bounded_uniform':
            strength = torch.ones(int(self.nonself.sum()))
        else:
            strength = attributes[self.nonself, 1].clone()
        if not bool(torch.isfinite(strength).all() and (strength > 0).all()):
            raise ValueError('Similarity priors must be finite and positive')
        self.register_buffer('prior_strength', strength)
        total = torch.zeros(n_target).index_add(0, self.destination, strength)
        fixed = torch.ones(edges.size(1))
        fixed[self.nonself] = strength * self.degree[self.destination] / total[self.destination].clamp_min(1e-12)
        self.fixed_weights = fixed

    def forward(self, x, edge_index, edge_attr=None, size=None):
        source, target = x if isinstance(x, (tuple, list)) else (x, x)
        if self.intervention == 'uniform':
            weights = source.new_ones(edge_index.size(1))
        elif self.intervention == 'prior_only':
            weights = self.fixed_weights
        else:
            s, d = edge_index[:, self.nonself]
            hidden = self.source(norm(source))[s] + self.target(norm(target))[d]
            if self.intervention != 'no_attributes':
                hidden = hidden + self.attribute(self.attributes[self.nonself])
            scores = self.scorer(F.leaky_relu(hidden, .2)).squeeze(-1)
            if self.mode == 'prior_free' and scores.numel():
                # exp(epsilon * score) matches the bounded arm's derivative
                # at zero but imposes no small multiplicative trust region.
                # A detached destination maximum gives stable exponentials;
                # its common factor cancels in the subsequent normalization.
                logits = self.epsilon * scores
                maxima = scatter(logits.detach(), self.destination, dim=0,
                                 dim_size=self.degree.numel(), reduce='max')
                multiplier = (logits - maxima[self.destination]).exp()
            else:
                multiplier = 1 + self.epsilon * scores.tanh()
            strength = (torch.ones_like(self.prior_strength)
                        if self.intervention == 'no_prior' else self.prior_strength)
            positive = strength * multiplier
            total = torch.zeros_like(self.degree).index_add(0, self.destination, positive)
            scaled = positive * self.degree[self.destination] / total[self.destination].clamp_min(1e-12)
            weights = source.new_ones(edge_index.size(1))
            weights[self.nonself] = scaled
        self.last_weights = weights.detach()
        return self.core((source, target), edge_index, edge_attr=weights, size=size)


class PriorResidualModel(EdgePathModel):
    def __init__(self, base, data, assets, config, arm, seed, width=16, epsilon=.25):
        super().__init__(base, data, assets, config, 'base', seed)
        self.arm = arm
        for layer in (0, 1):
            for i, (relation, attributes) in enumerate(assets['attributes'].items()):
                path = 'convs.' + str(layer) + '.' + '__'.join(relation)
                original = base.encoder.get_submodule(path)
                module = PriorResidualConv(original, data[relation].edge_index,
                                           attributes, data[relation[2]].num_nodes,
                                           arm, seed + 700003 + i, width, epsilon)
                parent, child = path.rsplit('.', 1)
                setattr(base.encoder.get_submodule(parent), child, module)
                self.edge_layers.append((path, module))


def make_model(base, data, assets, config, arm, seed, protocol):
    if arm in CONTROLS:
        return EdgePathModel(base, data, assets, config, arm, seed,
                             protocol['width'], protocol['edge_width'])
    return PriorResidualModel(base, data, assets, config, arm, seed,
                              protocol['edge_width'], protocol['epsilon'])


def interventions(arm):
    if arm == 'edge_attributes':
        return ('uniform', 'no_attributes')
    if arm in NEW_ARMS:
        return ('prior_only', 'no_prior', 'uniform')
    return ()
