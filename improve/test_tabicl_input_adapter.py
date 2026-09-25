"""Structural and optimization invariants for shared input adaptation."""
import numpy as np
from pathlib import Path
import tempfile
import torch
import unittest
from unittest.mock import patch
from scipy.optimize import minimize

from improve.tabicl_input_adapter_modules import (
    SharedInputAdapter, load_adapter, parameter_vector, project_update,
    ranking_losses, save_adapter,
    sample_episode,
)


def test_zero_recovery_filtering_and_semantic_permutations():
    torch.manual_seed(3)
    adapter = SharedInputAdapter()
    assert sum(p.numel() for p in adapter.parameters()) == 496
    retained = torch.tensor([0, 2, 6, 9, 12, 30, 31, 35])
    canonical = torch.randn(2, 17, len(retained))
    order = torch.stack([torch.randperm(len(retained)) for _ in range(2)])
    shuffled = canonical.gather(-1, order[:, None].expand_as(canonical))
    torch.testing.assert_close(adapter.transformed(shuffled, retained, order), shuffled, rtol=0, atol=0)
    with torch.no_grad():
        adapter.up.weight.normal_(std=.03)
    actual = adapter.transformed(shuffled, retained, order)
    expected = adapter.transformed(canonical, retained).gather(-1, order[:, None].expand_as(canonical))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    restored = actual.gather(-1, order.argsort(-1)[:, None].expand_as(actual))
    torch.testing.assert_close(restored[..., -2:], canonical[..., -2:], rtol=0, atol=0)
    assert (actual - shuffled).abs().max() > 0


def test_adapter_row_independence_and_persistence():
    torch.manual_seed(4)
    adapter = SharedInputAdapter()
    with torch.no_grad():
        adapter.up.weight.normal_(std=.01)
    x = torch.randn(1, 23, 36)
    indices = torch.tensor([8, 0, 8, 12, 7])
    torch.testing.assert_close(adapter(x[:, indices]), adapter(x)[:, indices])
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'adapter.pt'
        save_adapter(path, adapter, {'step': 64})
        restored, meta = load_adapter(path, 'cpu')
    torch.testing.assert_close(parameter_vector(restored), parameter_vector(adapter), rtol=0, atol=0)
    assert meta == {'step': 64}


def test_projection_agrees_with_independent_constrained_solver(seed):
    rng = np.random.default_rng(seed)
    proposed = rng.normal(size=9)
    gradients = rng.normal(size=(2, 9))
    actual = project_update(torch.tensor(proposed), torch.tensor(gradients)).numpy()
    expected = minimize(lambda d: .5 * np.sum((d - proposed) ** 2), np.zeros(9),
        jac=lambda d: d - proposed, method='SLSQP',
        constraints={'type': 'ineq', 'fun': lambda d: -(gradients @ d), 'jac': lambda d: -gradients},
        options={'ftol': 1e-12, 'maxiter': 200})
    assert expected.success
    assert np.max(gradients @ actual) <= 1e-10
    np.testing.assert_allclose(actual, expected.x, atol=1e-7, rtol=1e-7)


def test_opposed_and_zero_constraints():
    proposal = torch.tensor([2., 3., 4.], dtype=torch.float64)
    gradients = torch.tensor([[1., 0., 0.], [-1., 0., 0.]], dtype=torch.float64)
    torch.testing.assert_close(project_update(proposal, gradients), torch.tensor([0., 3., 4.], dtype=torch.float64))
    torch.testing.assert_close(project_update(proposal, torch.zeros_like(gradients)), proposal)


def test_project_actual_adam_delta_instead_of_raw_gradient():
    # Momentum can produce a harmful positive change although -gradient is safe.
    parameter = torch.nn.Parameter(torch.tensor([0., 0.]))
    optimizer = torch.optim.Adam([parameter], lr=.1)
    for _ in range(4):
        parameter.grad = torch.tensor([-1., 0.])
        optimizer.step()
    before = parameter.detach().clone()
    parameter.grad = torch.tensor([.01, 0.])
    assert -parameter.grad[0] < 0
    optimizer.step()
    actual_delta = parameter.detach() - before
    assert actual_delta[0] > 0
    projected = project_update(actual_delta, torch.tensor([[1., 0.], [0., 1.]]))
    assert projected[0] <= 1e-10


def test_ranking_objectives_direction_shift_and_gradients():
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    margin = torch.tensor([-3., 3., -2., 4., -4., 2.], requires_grad=True)
    good = ranking_losses(margin, labels)
    bad = ranking_losses(-margin, labels)
    assert torch.all(good < bad)
    torch.testing.assert_close(ranking_losses(margin + 10., labels), good)
    good.sum().backward()
    assert torch.isfinite(margin.grad).all()
    assert margin.grad[labels == 1].max() < 0
    assert margin.grad[labels == 0].min() > 0


def test_stratified_episode_is_disjoint_and_preserves_prevalence():
    labels = np.r_[np.ones(30, dtype=int), np.zeros(70, dtype=int)]
    support, query = sample_episode(labels, np.arange(100), 24, 8, np.random.default_rng(1))
    assert len(support) == 24 and len(query) == 8
    assert len(np.intersect1d(support, query)) == 0
    assert labels[support].sum() == 7 and labels[query].sum() == 2


def test_historical_prediction_schema_and_view_alignment():
    from improve.tabicl_input_adapter import parent_predictions
    labels = np.array([0, 1, 0])
    pairs = np.array([[0, 1, 2], [4, 4, 5]])
    views = np.arange(18).reshape(6, 3) / 18.
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'scores.npz'
        np.savez(path, views=views, y_val=labels, pairs_val=pairs)
        np.testing.assert_array_equal(parent_predictions(temporary, 'LncDrug', 'full', labels, pairs), views)
        np.testing.assert_array_equal(parent_predictions(temporary, 'MiDrug', 'full', labels, pairs), views[:1])
        np.savez(path, views=views[:2], y_val=labels, pairs_val=pairs)
        np.testing.assert_array_equal(parent_predictions(temporary, 'MiDrug', 'fold0', labels, pairs), views[1:2])


def test_outer_features_cannot_load_before_own_selection():
    from improve import tabicl_input_adapter as runner
    with tempfile.TemporaryDirectory() as temporary, patch.object(runner.graph, 'features') as loader:
        try:
            runner.stage_arrays(temporary, {'sha256': 'unselected', 'smoke': False}, 'MiDrug', 'outer')
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('Unselected trial reached outer data')
        loader.assert_not_called()


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for function in [test_zero_recovery_filtering_and_semantic_permutations,
        test_adapter_row_independence_and_persistence, test_opposed_and_zero_constraints,
        test_project_actual_adam_delta_instead_of_raw_gradient,
        test_stratified_episode_is_disjoint_and_preserves_prevalence,
        test_historical_prediction_schema_and_view_alignment,
        test_outer_features_cannot_load_before_own_selection,
        test_ranking_objectives_direction_shift_and_gradients]:
        suite.addTest(unittest.FunctionTestCase(function))
    for seed in range(6):
        suite.addTest(unittest.FunctionTestCase(
            lambda seed=seed: test_projection_agrees_with_independent_constrained_solver(seed),
            description='Independent constrained-solver agreement, seed %d' % seed))
    return suite


if __name__ == '__main__':
    unittest.main()
