"""Class-conditional risk distribution matching on training-only drug domains.

An independent graph adaptation of Nguyen et al., WACV 2024. The domain
partition is used only by the training objective, never as a prediction input.
"""
import numpy as np
from sklearn.cluster import KMeans
import torch

from improve import overlap_block_models as graph_lib
from improve import sharpness_graph_v1 as backbone
from improve.nested_anchor_inputs_v2 import TRAIN_KEYS as INPUT_KEYS

KINDS = ('full', 'moment_matching', 'erm')
identity_task = backbone.identity_task


def parameters():
    return dict(**backbone.parameters(), domain_count=4, domain_seed=20260916,
                domain_kmeans_n_init=10, warmup_steps=100, matching_weight=1.,
                kernel_gammas=[.0001, .001, .01, .1, 1., 10., 100., 1000.],
                kernel_reduction='sum', class_conditional=True,
                minimum_class_domain_samples=2)


def drug_domains(a, p):
    """Cluster only drug attributes of entities present in training pairs."""
    pairs = np.asarray(a['pairs_train'])
    used = np.unique(pairs[1])
    nd = int(a['nd'])
    if (len(used) < p['domain_count'] or used.min() < 0 or used.max() >= nd
            or a['x_2'].shape[0] != nd):
        raise ValueError('Insufficient or invalid training drug identities')
    x = np.asarray(a['x_2'][used], dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite drug attributes')
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
    labels = KMeans(n_clusters=p['domain_count'], n_init=p['domain_kmeans_n_init'],
                    random_state=p['domain_seed'], algorithm='lloyd').fit_predict(x)
    if len(np.unique(labels)) != p['domain_count']:
        raise ValueError('Degenerate drug-domain partition')
    domains = np.full(nd, -1, dtype=np.int64)
    domains[used] = labels
    return domains


def rbf_mmd(risks, difference_weights, gammas):
    """Exact weighted empirical MMD squared, including diagonal terms.

    The kernel is the sum of eight RBFs, as in the pinned DomainBed code.
    No Monte Carlo estimate, detached risk, or fitted bandwidth is used.
    """
    if risks.ndim != 1 or difference_weights.shape != risks.shape:
        raise ValueError('Expected scalar risks and aligned distribution weights')
    if not gammas or any(float(g) <= 0 for g in gammas):
        raise ValueError('RBF inverse bandwidths must be positive')
    if abs(float(difference_weights.sum())) > 1e-8:
        raise ValueError('Difference of probability weights must sum to zero')
    distances = (risks[:, None] - risks[None, :]).square()
    result = risks.sum() * 0.
    for gamma in gammas:
        kernel = torch.exp(-float(gamma) * distances)
        result = result + torch.dot(difference_weights, kernel @ difference_weights)
    return result


def matching_terms(risks, labels, domains, p):
    """Match each label's worst-mean-risk domain to its equal-domain mixture.

    All samples are actual training pairs from the current omitted support
    fold. The matching is conditional on their training labels so different
    class proportions alone do not create a domain penalty. This conditioning
    and the weighted treatment of unequal domain sizes are local adaptations.
    """
    if (risks.ndim != 1 or risks.shape != labels.shape or domains.shape != risks.shape
            or not torch.isfinite(risks).all() or not torch.all((labels == 0) | (labels == 1))
            or bool((domains < 0).any()) or not p['class_conditional']):
        raise ValueError('Invalid class-conditional training risks')
    distribution, moment, records = [], [], []
    for label in (0, 1):
        selected = labels == label
        values, groups = risks[selected], domains[selected]
        unique, inverse, counts = torch.unique(groups, sorted=True, return_inverse=True, return_counts=True)
        if len(unique) < 2 or bool((counts < p['minimum_class_domain_samples']).any()):
            raise ValueError('Insufficient class-domain samples in training fold')
        means = torch.stack([values[groups == domain].mean() for domain in unique])
        # Selection is piecewise constant; gradients still reach all risks in
        # the chosen distribution and in the complete mixture.
        worst_index = int(means.detach().argmax())
        worst = unique[worst_index]
        mixture = 1. / (len(unique) * counts[inverse].to(values.dtype))
        selected_domain = (groups == worst).to(values.dtype) / counts[worst_index]
        difference = selected_domain - mixture
        distribution.append(rbf_mmd(values, difference, p['kernel_gammas']))
        mean_all = torch.dot(mixture, values)
        mean_worst = means[worst_index]
        variance_all = torch.dot(mixture, (values - mean_all).square())
        variance_worst = torch.dot(selected_domain, (values - mean_worst).square())
        moment.append((mean_worst - mean_all).square() + (variance_worst - variance_all).square())
        records.append(dict(label=label, domains=unique.detach().cpu().tolist(),
                            counts=counts.detach().cpu().tolist(),
                            mean_risks=means.detach().cpu().tolist(),
                            worst_domain=int(worst)))
    return torch.stack(distribution).mean(), torch.stack(moment).mean(), records


def setup(a, p, kind, device):
    if set(a) - (INPUT_KEYS | {'anchor_oof'}):
        raise ValueError('Risk-domain setup accepts training arrays only')
    if kind not in KINDS:
        raise ValueError('Unregistered risk-matching arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        return backbone.IdentityResidual().to(device), [], [None] * p['folds'], folds
    n, relations = int(a['node_counts'].sum()), int(a['relation'].max()) + 1
    graphs = []
    for f in range(p['folds']):
        edges, relation = graph_lib.masked_physical(a, np.flatnonzero(a['folds'] == f))
        graphs.append(graph_lib.graph_inputs(edges, relation, np.empty((2, 0), dtype=np.int64),
                                            n, relations, device))
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = backbone.OrdinaryGraph([x.shape[1] for x in xs], relations, p).to(device)
    model.register_buffer('drug_domains', torch.as_tensor(drug_domains(a, p), device=device))
    return model, xs, graphs, folds
