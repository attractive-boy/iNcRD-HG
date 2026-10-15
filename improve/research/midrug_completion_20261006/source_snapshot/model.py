"""Convex observed-entry logistic matrix completion, with optional sparse errors.

Objective: sum_i min_e BCE(y_i, b + X_i + e) + tau*abs(e)
           + lambda*||X||_* + gamma/2*(tr(X.T Lr X)+tr(X Ld X.T)).
tau=None disables e. Only observed training entries contribute supervised loss.
"""
import numpy as np
from scipy import linalg, sparse
from scipy.optimize import brentq
from scipy.special import expit

CONFIGS = {
    f'n{int(frac*100):02d}_g{int(gamma*100):02d}_{mode}':
        dict(nuclear_fraction=frac, graph_weight=gamma, error_cost=tau)
    for frac in (.03, .1) for gamma in (0., .05)
    for mode, tau in [('logistic', None), ('robust', .25)]
}


def laplacian(n, edges):
    """Symmetric graph topology, combinatorial Laplacian normalized to norm 1."""
    row, col = np.asarray(edges)
    w = sparse.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    w = w.maximum(w.T)
    w.data[:] = 1
    w.setdiag(0)
    w.eliminate_zeros()
    lap = sparse.diags(np.asarray(w.sum(1)).ravel())-w
    norm = float(linalg.eigvalsh(lap.toarray(), subset_by_index=[n-1, n-1])[0])
    return (lap/max(norm, 1e-12)).tocsr()


def observed_loss_gradient(z, y, tau):
    sign = 2*y-1
    margin = sign*z
    if tau is None:
        return float(np.logaddexp(0., -margin).sum()), expit(z)-y
    assert 0 < tau < .5
    boundary = np.log((1-tau)/tau)
    delta = np.maximum(boundary-margin, 0.)
    loss = np.logaddexp(0., -np.maximum(margin, boundary))+tau*delta
    grad = -sign*np.minimum(expit(-margin), tau)
    return float(loss.sum()), grad


def intercept(y, tau):
    prior = float(np.mean(y))
    nominal = np.log(prior/(1-prior))
    if tau is None:
        return nominal, nominal
    fitted = brentq(lambda b: observed_loss_gradient(np.full(len(y), b), y, tau)[1].sum(), -30., 30.)
    return float(fitted), nominal


def shrink(x, threshold):
    u, s, v = linalg.svd(x, full_matrices=False, check_finite=False, lapack_driver='gesdd')
    s = np.maximum(s-threshold, 0.)
    keep = s > 0
    return (u[:, keep]*s[keep])@v[keep], float(s.sum()), int(keep.sum())


def smooth(x, pairs, y, b, tau, lr, ld, gamma):
    row, col = pairs.T
    obs, og = observed_loss_gradient(b+x[row, col], y, tau)
    gradient = np.zeros_like(x)
    gradient[row, col] = og
    graph = 0.
    if gamma:
        gx = gamma*(lr@x+(ld@x.T).T)
        graph = .5*float(np.sum(x*gx))
        gradient += gx
    return obs+graph, gradient, dict(observed_loss=obs, graph_penalty=graph)


def fit(pairs, y, lr, ld, cfg, max_steps=3000, tolerance=1e-5, callback=None):
    """Training-only stopping; validation data are never accepted by this API."""
    assert len(np.unique(pairs, axis=0)) == len(pairs)
    b, nominal = intercept(y, cfg['error_cost'])
    shape = (lr.shape[0], ld.shape[0])
    x = np.zeros(shape, dtype=np.float64)
    _, initial_grad, _ = smooth(x, pairs, y, b, cfg['error_cost'], lr, ld, 0.)
    lam_max = float(linalg.svdvals(initial_grad, check_finite=False)[0])
    lam = cfg['nuclear_fraction']*lam_max
    gamma = cfg['graph_weight']
    step = 1/(.25+2*gamma)
    momentum = x.copy()
    acceleration = 1.
    previous_obj = float('inf')
    curves = []
    converged = False
    for iteration in range(1, max_steps+1):
        _, grad, _ = smooth(momentum, pairs, y, b, cfg['error_cost'], lr, ld, gamma)
        proposal, nuclear, rank = shrink(momentum-step*grad, step*lam)
        value, _, parts = smooth(proposal, pairs, y, b, cfg['error_cost'], lr, ld, gamma)
        objective = value+lam*nuclear
        restarted = objective > previous_obj+1e-9
        if restarted:
            acceleration = 1.
            _, grad, _ = smooth(x, pairs, y, b, cfg['error_cost'], lr, ld, gamma)
            proposal, nuclear, rank = shrink(x-step*grad, step*lam)
            value, _, parts = smooth(proposal, pairs, y, b, cfg['error_cost'], lr, ld, gamma)
            objective = value+lam*nuclear
        assert np.isfinite(objective) and objective <= previous_obj+1e-7
        next_acceleration = (1+np.sqrt(1+4*acceleration**2))/2
        momentum = proposal+(acceleration-1)/next_acceleration*(proposal-x)
        x = proposal
        acceleration = next_acceleration
        previous_obj = objective
        if iteration == 1 or iteration % 50 == 0 or iteration == max_steps:
            _, grad, _ = smooth(x, pairs, y, b, cfg['error_cost'], lr, ld, gamma)
            prox, _, _ = shrink(x-step*grad, step*lam)
            residual = float(linalg.norm(x-prox)/step/max(1., linalg.norm(x)))
            record = dict(iteration=iteration, objective=objective, nuclear_penalty=lam*nuclear,
                          rank=rank, stationarity_residual=residual, **parts)
            curves.append(record)
            if callback:
                callback(record)
            if residual <= tolerance:
                converged = True
                break
    return x, dict(intercept=b, nominal_intercept=nominal, lambda_max=lam_max, nuclear_weight=lam,
                   step_size=step, iterations=iteration, converged=converged, curves=curves)
