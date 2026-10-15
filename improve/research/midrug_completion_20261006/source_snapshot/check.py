"""Check the actual optimization contract on small independently solvable cases."""
import json
import numpy as np
from scipy import sparse
from scipy.optimize import minimize_scalar
from common import HERE, load_inner, make_graph
from model import observed_loss_gradient, laplacian, shrink, fit, smooth, intercept


def main():
    rng = np.random.default_rng(42)
    z, y = rng.normal(size=30)*3, rng.integers(0, 2, 30).astype(float)
    records = []
    for tau in (None, .25):
        value, gradient = observed_loss_gradient(z, y, tau)
        numeric = []
        for i in range(len(z)):
            plus, minus = z.copy(), z.copy()
            plus[i] += 1e-5
            minus[i] -= 1e-5
            numeric.append((observed_loss_gradient(plus, y, tau)[0]-observed_loss_gradient(minus, y, tau)[0])/2e-5)
        np.testing.assert_allclose(gradient, numeric, atol=2e-8)
        if tau:
            brute = sum(minimize_scalar(lambda e: np.logaddexp(0., -(2*yi-1)*(zi+e))+tau*abs(e),
                                       bounds=(-30,30), method='bounded').fun for zi, yi in zip(z,y))
            assert abs(brute-value) < 2e-5
        b, _ = intercept(y, tau)
        assert abs(observed_loss_gradient(np.full(len(y), b), y, tau)[1].sum()) < 1e-10
        records.append(dict(error_cost=tau, finite_difference_error=float(np.max(abs(gradient-numeric)))))
    out, nuclear, rank = shrink(np.diag([4.,2.,.5]), 1.)
    np.testing.assert_allclose(out, np.diag([3.,1.,0.]), atol=1e-12)
    assert nuclear == 4. and rank == 2
    # Fully observed 2x2 case has a transparent low-rank optimum.
    pairs = np.array([[0,0],[0,1],[1,0],[1,1]])
    labels = np.array([1.,0.,0.,1.])
    zero = sparse.csr_matrix((2,2))
    x, info = fit(pairs, labels, zero, zero, dict(error_cost=None,nuclear_fraction=.1,graph_weight=0.), tolerance=1e-9)
    assert info['converged']
    np.testing.assert_allclose(x, np.array([[1,-1],[-1,1]])*np.log(19), atol=1e-6)
    # Unobserved cells make zero direct likelihood contribution and gradient.
    v, g, _ = smooth(np.zeros((2,2)), pairs[:2], labels[:2], 0., None, zero, zero, 0.)
    assert np.count_nonzero(g[1]) == 0
    data = load_inner(0)
    graph = make_graph(data)
    assert set(graph.node_types) == {'miRNA','drug'}
    for name, rel in [('miRNA','MiMi'),('drug','DrugDrug')]:
        lap = laplacian(graph[name].num_nodes, graph[name, rel, name].edge_index.numpy())
        np.testing.assert_allclose(lap.toarray(), lap.T.toarray(), atol=1e-12)
        np.testing.assert_allclose(lap@np.ones(lap.shape[0]), 0., atol=1e-12)
        eigen = np.linalg.eigvalsh(lap.toarray())
        assert eigen.min() > -1e-10 and eigen.max() <= 1+1e-10
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed', records=records,
        scalar_sparse_error_equivalence=True, exact_2x2_solution=True, zero_unobserved_likelihood=True,
        symmetric_psd_laplacians=True, only_mirna_drug=True), indent=2))
    print('Initial checks passed', flush=True)


if __name__ == '__main__':
    main()
