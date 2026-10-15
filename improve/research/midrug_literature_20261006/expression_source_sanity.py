"""Synthetic checks of selected source claims; no datasets or pretrained weights."""
import ast
import hashlib
import json
import math
from pathlib import Path
import torch
from torch import nn

HERE = Path(__file__).resolve().parent


def main():
    source = HERE / 'raw/expression_bes_author_model.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == 'MultiHeadSelfAttention')
    # Execute only the self-contained attention class, never the author loader.
    namespace = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    torch.manual_seed(42)
    layer = namespace['MultiHeadSelfAttention'](8, 2).double()
    x = torch.randn(5, 1, 8, dtype=torch.double)
    y = layer(x)
    reference = layer.fc(layer.value(x)) + x
    torch.testing.assert_close(y, reference, atol=1e-12, rtol=1e-12)
    y.square().sum().backward()
    zero_grad = {name: float(p.grad.abs().max())
                 for name, p in layer.named_parameters() if name.startswith(('query.', 'key.'))}
    assert all(v == 0 for v in zero_grad.values())
    layer.zero_grad()
    layer(torch.randn(5, 2, 8, dtype=torch.double)).square().sum().backward()
    nonzero = {name: float(p.grad.abs().max()) for name, p in layer.named_parameters()
               if name.startswith(('query.', 'key.'))}
    assert nonzero['query.weight'] > 0 and nonzero['key.weight'] > 0

    # ReCoG Eq. (4) can weaken an existing unit edge when s < 1.
    a, similarity, alpha = 1., 0., .5
    refined = alpha * a + (1 - alpha) * similarity
    assert refined < a

    # The squared-distance sum identity does not imply log-sum-exp equivalence.
    point = torch.tensor(.3, dtype=torch.double, requires_grad=True)
    members = torch.tensor([-1., 1.], dtype=torch.double)
    exact_sum = (point - members).square().sum()
    center_sum = 2 * point.square() + members.square().sum()
    sum_g = torch.autograd.grad(exact_sum, point, retain_graph=True)[0]
    center_g = torch.autograd.grad(center_sum, point, retain_graph=True)[0]
    torch.testing.assert_close(sum_g, center_g)
    lse = torch.logsumexp(-(point - members).square(), 0)
    center_lse = math.log(2.) - point.square()
    lse_g = torch.autograd.grad(lse, point)[0]
    center_lse_g = torch.autograd.grad(center_lse, point)[0]
    assert abs(lse_g - center_lse_g) > .1
    result = dict(
        status='passed', no_datasets_loaded=True, no_pretrained_weights_loaded=True,
        bes_author_commit='73bccf712544016c19cff512f69038a98d1d11c4',
        bes_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        singleton_attention=dict(affine_error=float(abs(y-reference).max()),
                                 qk_gradient=zero_grad, two_token_control_gradient=nonzero),
        recog_eq4=dict(original=a, similarity=similarity, alpha=alpha, refined=refined),
        centroid_identity=dict(pairwise_squared_sum_gradient=float(sum_g),
                               centroid_squared_sum_gradient=float(center_g),
                               pairwise_logsumexp_gradient=float(lse_g),
                               centroid_logsumexp_gradient=float(center_lse_g)),
        scope='Synthetic local checks only; not a reproduction or an audit of all paper results.')
    (HERE / 'expression_source_sanity.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
