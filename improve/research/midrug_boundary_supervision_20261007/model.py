"""Training-only pair boundary sampling and stopped-gradient prototype supervision."""
import math
import torch
from torch.nn import functional as F
from common import Adapter, make_graph, Reconstruction

CONFIGS = {name + suffix: dict(selection=name, fraction=.25, ridge_fraction=.1,
           std_floor=.001, temperature=.2, auxiliary_weight=.05, reconstruction_weight=weight)
           for name in ['boundary', 'random'] for suffix, weight in [('', 0.), ('_rec10', 10.)]}


def make_adapter(data, name, device):
    a = Adapter(data, make_graph(data), device)
    if name == 'unmasked10' or CONFIGS.get(name, {}).get('reconstruction_weight', 0):
        a.model.add_module('reconstruction', Reconstruction(a.graph, a.cfg.hidden).to(device))
    return a


def mask_generator(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


def make_optimizer(m):
    groups = [dict(params=[p for n, p in m.named_parameters() if not n.startswith('reconstruction.')], lr=1e-4)]
    if hasattr(m, 'reconstruction'):
        groups.append(dict(params=list(m.reconstruction.parameters()), lr=1e-4, weight_decay=1e-4))
    return torch.optim.RMSprop(groups, alpha=.99)


def capture_features(a, part='train'):
    captured = []
    hook = a.model.lin3.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        logits = a.logits(part)
    finally:
        hook.remove()
    assert len(captured) == 1
    return logits, captured[0]


@torch.no_grad()
def training_context(z, y, cfg, rng):
    """Only training pair features/labels. Both arms compute the same two masks.

    Geometry is computed in float64. Each class supplies ceil(fraction*n_class)
    anchors, so boundary and random arms have exactly equal budgets at every
    epoch even after their learned representations diverge.
    """
    assert z.ndim == 2 and y.shape == (len(z),) and set(y.unique().tolist()) == {0., 1.}
    x = z.detach().double()
    labels = y.long()
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(cfg['std_floor'])
    standardized = (x-mean)/scale
    centered = standardized-standardized.mean(0)
    covariance = centered.T@centered/(len(x)-1)
    ridge = cfg['ridge_fraction']*covariance.diag().mean().clamp_min(1e-8)
    covariance = covariance+ridge*torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    centers = torch.stack([standardized[labels == c].mean(0) for c in [0, 1]])
    direction = torch.linalg.solve(covariance, centers[1]-centers[0])
    midpoint = .5*(centers[0]+centers[1])
    margin = ((standardized-midpoint)@direction).abs()
    boundary, random, counts = [], [], []
    for c in [0, 1]:
        indices = torch.where(labels == c)[0]
        count = math.ceil(cfg['fraction']*len(indices))
        counts.append(count)
        # Stable tie-break: original training record order.
        boundary.append(indices[torch.argsort(margin[indices], stable=True)[:count]])
        random.append(indices[torch.randperm(len(indices), generator=rng, device=x.device)[:count]])
    unit = F.normalize(standardized, dim=1, eps=1e-12)
    prototypes = F.normalize(torch.stack([unit[labels == c].mean(0) for c in [0, 1]]), dim=1, eps=1e-12)
    chosen = torch.cat(boundary if cfg['selection'] == 'boundary' else random)
    context = dict(mean=mean.to(z.dtype), scale=scale.to(z.dtype), prototypes=prototypes.to(z.dtype),
                   selected=chosen, boundary=torch.cat(boundary), random=torch.cat(random), counts=counts,
                   margin=margin, ridge=ridge, direction=direction)
    assert all(not value.requires_grad for value in context.values() if isinstance(value, torch.Tensor))
    return context


def prototype_loss(z, y, context, temperature):
    """Gradients affect selected anchors; center/scale/prototypes remain fixed."""
    indices = context['selected']
    unit = F.normalize((z[indices]-context['mean'])/context['scale'], dim=1, eps=1e-12)
    scores = unit@context['prototypes'].T/temperature
    return F.cross_entropy(scores, y[indices].long())


def association_loss(a, cfg, epoch, rng):
    logits, z = capture_features(a)
    bce = F.binary_cross_entropy_with_logits(logits, a.y)
    weight = cfg['auxiliary_weight']*min(1., max(0., (epoch-30)/30.))
    auxiliary = bce.new_zeros(())
    context = None
    counts, mean_margin, overlap = [0, 0], 0., 0
    if weight:
        context = training_context(z, a.y, cfg, rng)
        auxiliary = prototype_loss(z, a.y, context, cfg['temperature'])
        counts = context['counts']
        mean_margin = float(context['margin'][context['selected']].mean())
        overlap = int(torch.isin(context['boundary'], context['random']).sum())
    total = bce+weight*auxiliary
    stats = dict(association_loss=float(total.detach()), clean_bce=float(bce.detach()),
                 auxiliary=float(auxiliary.detach()), auxiliary_weight=weight,
                 selected_zero_count=counts[0], selected_positive_count=counts[1],
                 selected_mean_abs_margin=mean_margin, random_boundary_overlap=overlap)
    return total, stats, dict(logits=logits, features=z, context=context, clean=bce, auxiliary=auxiliary)
