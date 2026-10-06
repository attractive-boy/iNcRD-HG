"""Run exact selected BEC functions without its unavailable PyG dependency."""
import ast
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    output = HERE / 'BEC_AUTHOR_AUDIT_V2.json'
    if output.exists():
        raise ValueError('Preserve previous BEC audit')
    source = HERE / 'bec_author_v1_Vertex_classification__models.py'
    manifest = base.read_json(HERE / 'BEC_AUTHOR_SOURCE_V1.json')
    item = next(row for row in manifest if row['path'] == 'Vertex_classification/models.py')
    assert base.digest(source) == item['sha256']
    tree = ast.parse(source.read_text())
    names = {'compute_gamma', 'compute_gamma2_optimized', 'compute_loss', 'MLP', 'CurvatureMLP'}
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    gnn = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DC_GraphSAGE')
    mask_function = next(node for node in gnn.body if isinstance(node, ast.FunctionDef)
                         and node.name == 'remove_top_k_neighbors_with_mask')
    selected.append(mask_function)
    namespace = dict(torch=torch, nn=nn, F=F)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    torch.set_num_threads(2)
    torch.random.default_generator.manual_seed(11603)
    a = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64)
    f = torch.tensor([0., 1.], dtype=torch.float64)
    gamma = namespace['compute_gamma'](f, a, torch.ones_like(a))
    gamma2 = namespace['compute_gamma2_optimized'](f, a, torch.ones_like(a))
    torch.testing.assert_close(gamma, torch.full((2,), .5, dtype=torch.float64), atol=0., rtol=0.)
    torch.testing.assert_close(gamma2, torch.ones(2, dtype=torch.float64), atol=0., rtol=0.)
    kappa = torch.tensor([[.1], [.5], [.9]], dtype=torch.float64)
    g, g2 = torch.tensor([1., 2., 3.], dtype=torch.float64), torch.tensor([.2, 1., 2.], dtype=torch.float64)
    actual = namespace['compute_loss'](kappa, g, g2)
    expected = F.relu(kappa.flatten() * g - g2).sum()
    assert actual != expected
    plain_list = any(isinstance(node, ast.Assign) and isinstance(node.value, ast.List)
        and any(isinstance(target, ast.Attribute) and target.attr == 'fn_mlp' for target in node.targets)
        for node in ast.walk(gnn))
    assert plain_list
    model = namespace['CurvatureMLP'](3, 4)
    curvature = model(torch.randn(4, 3))
    adj = torch.ones(4, 4) - torch.eye(4)
    mask = namespace['remove_top_k_neighbors_with_mask'](None, adj, curvature, 25)
    latent = torch.randn(4, 3, requires_grad=True)
    surrogate_task = (mask @ latent).square().sum()
    gradients = torch.autograd.grad(surrogate_task, list(model.parameters()), allow_unused=True)
    assert all(value is None for value in gradients)
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=base.digest(source), author_commit=item['commit'], script_sha256=base.digest(__file__),
        previous_full_module_import_failure_sha256=base.digest(HERE / 'bec_author_audit_v1.log'),
        previous_failure='torch_geometric is not installed in the frontier venv',
        execution_scope='Exact AST functions/classes listed below; no full GNN import or training',
        selected_definitions=sorted(names) + [mask_function.name], author_ast_modified=False,
        two_node_example=dict(gamma=gamma.tolist(), code_gamma2=gamma2.tolist(),
            definition_gamma2=[1., 1.], printed_expanded_formula_gamma2=[2., 2.],
            code_matches_operator_definition=True, printed_expansion_missing_one_half=True),
        loss_broadcast=dict(kappa_shape=list(kappa.shape), gamma_shape=list(g.shape),
            broadcast_shape=list((kappa * g).shape), code_loss=float(actual), vertexwise_loss=float(expected)),
        registration_static_audit=dict(class_name='DC_GraphSAGE', fn_mlp_is_plain_list=plain_list,
            full_module_registration_not_executed=True),
        mask_gradient=dict(exact_author_mask_used=True, downstream_test='squared aggregate of independent latent features',
            all_curvature_parameter_gradients_none=True, full_author_GNN_task_not_executed=True),
        model_fitting_performed=False, biological_inputs_read=False, evaluation_metrics_computed=False,
        decision='Do not run the selected released path as a faithful task-conditioned curvature estimator; no local performance conclusion',
        confirmed_final_innovations=0))
    base.write_json(output, result)
    print('BEC_AUTHOR_AST_AUDIT', result['loss_broadcast'], result['mask_gradient'], flush=True)


if __name__ == '__main__':
    main()
