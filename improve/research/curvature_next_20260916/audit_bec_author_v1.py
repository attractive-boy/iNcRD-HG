"""Read-only synthetic audit of the pinned BEC vertex-classification path."""
from datetime import datetime, timezone
import importlib.util
from pathlib import Path

import torch
from torch.nn import functional as F

from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    output = HERE / 'BEC_AUTHOR_AUDIT_V1.json'
    if output.exists():
        raise ValueError('Preserve original author-code audit')
    source = HERE / 'bec_author_v1_Vertex_classification__models.py'
    manifest = base.read_json(HERE / 'BEC_AUTHOR_SOURCE_V1.json')
    item = next(row for row in manifest if row['path'] == 'Vertex_classification/models.py')
    assert base.digest(source) == item['sha256']
    spec = importlib.util.spec_from_file_location('pinned_bec_vertex_v1', source)
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    torch.set_num_threads(2)
    torch.random.default_generator.manual_seed(11603)

    adjacency = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64)
    values = torch.tensor([0., 1.], dtype=torch.float64)
    gamma = author.compute_gamma(values, adjacency, torch.ones_like(adjacency))
    gamma2 = author.compute_gamma2_optimized(values, adjacency, torch.ones_like(adjacency))
    torch.testing.assert_close(gamma, torch.full((2,), .5, dtype=torch.float64), atol=0., rtol=0.)
    torch.testing.assert_close(gamma2, torch.ones(2, dtype=torch.float64), atol=0., rtol=0.)

    kappa = torch.tensor([[.1], [.5], [.9]], dtype=torch.float64)
    g = torch.tensor([1., 2., 3.], dtype=torch.float64)
    g2 = torch.tensor([.2, 1., 2.], dtype=torch.float64)
    implemented_loss = author.compute_loss(kappa, g, g2)
    vertexwise_loss = F.relu(kappa.flatten() * g - g2).sum()
    assert implemented_loss != vertexwise_loss

    adj = torch.ones(4, 4) - torch.eye(4)
    model = author.DC_GraphSAGE(3, 4, 2, 0., 2, adj, fn_count=2)
    model.eval()
    parameter_ids = {id(value) for value in model.parameters()}
    missing = sum(id(value) not in parameter_ids for fn in model.fn_mlp for value in fn.parameters())
    assert missing == sum(1 for fn in model.fn_mlp for _ in fn.parameters()) > 0
    prediction, curvature_loss = model(torch.randn(4, 3), adj, p=25)
    target = torch.tensor([0, 1, 0, 1])
    parameters = list(model.curv_mlp.parameters()) + [model.weights]
    gradients = torch.autograd.grad(F.nll_loss(prediction, target), parameters, allow_unused=True, retain_graph=True)
    assert all(gradient is None for gradient in gradients)
    total_gradients = torch.autograd.grad(F.nll_loss(prediction, target) + curvature_loss,
        parameters, allow_unused=True)
    assert any(gradient is not None for gradient in total_gradients)

    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=base.digest(source), author_commit=item['commit'], script_sha256=base.digest(__file__),
        scope='Pinned Vertex_classification/models.py; synthetic DC_GraphSAGE path, not all repository variants',
        two_node_example=dict(gamma=gamma.tolist(), code_gamma2=gamma2.tolist(),
            definition_gamma2=[1., 1.], printed_expanded_formula_gamma2=[2., 2.],
            code_matches_operator_definition=True, printed_expansion_missing_one_half=True),
        loss_broadcast=dict(kappa_shape=list(kappa.shape), gamma_shape=list(g.shape),
            broadcast_shape=list((kappa * g).shape), code_loss=float(implemented_loss),
            vertexwise_loss=float(vertexwise_loss), author_loss_does_not_match_vertexwise_penalty=True),
        registration=dict(function_network_parameter_tensors=missing, all_absent_from_model_parameters=True,
            function_modules_stored_in_plain_list=True),
        task_gradient=dict(curvature_parameter_tensors=len(parameters), task_only_gradients_all_none=True,
            auxiliary_loss_supplies_gradients=True, hard_rank_and_nonzero_indices=True),
        model_fitting_performed=False, biological_inputs_read=False, evaluation_metrics_computed=False,
        decision='Do not execute this as a faithful task-conditioned curvature estimator without an explicit corrected algorithm and new checks',
        confirmed_final_innovations=0))
    base.write_json(output, result)
    print('BEC_AUTHOR_AUDIT', result['loss_broadcast'], result['registration'], result['task_gradient'], flush=True)


if __name__ == '__main__':
    main()
