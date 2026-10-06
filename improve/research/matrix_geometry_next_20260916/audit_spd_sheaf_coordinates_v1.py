"""Synthetic audit of two pinned coordinate routines and flat log coordinates."""
import ast
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.linalg
import torch

from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    destination = HERE / 'SPD_SHEAF_COORDINATE_AUDIT_V1.json'
    if destination.exists():
        raise ValueError('Preserve prior source audit')
    records = base.read_json(HERE / 'SPD_SHEAF_AUTHOR_SOURCE_V1.json')
    for row in records:
        assert row['git_blob_verified'] and base.digest(row['file']) == row['sha256']
        assert row['commit'] == '2c21b53efe9ce1f078c305bd9a4f3ecda57a9833'
    row = next(row for row in records if row['path'] == 'model/SPDsheaf.py')
    tree = ast.parse(Path(row['file']).read_text())
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SPDGNNModel')
    names = ('_construct_coordinate_spd', 'generate_equivariant_orthogonal_matrix')
    methods = [node for node in model.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(methods) == 2
    scope = dict(torch=torch)
    exec(compile(ast.Module(body=methods, type_ignores=[]), row['file'], 'exec'), scope)
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    generator = torch.Generator().manual_seed(4319)
    pos = torch.randn(13, 3, generator=generator)
    pos -= pos.mean(0)
    src = torch.arange(13)
    dst = (src + 1) % 13
    obj = SimpleNamespace(epsilon=.001, coord_method='outer_product')
    matrix = scope[names[0]](obj, pos, pos)
    frames = scope[names[1]](obj, pos, src, dst)
    identity = torch.eye(3).expand(13, -1, -1)
    torch.testing.assert_close(frames.transpose(-1, -2) @ frames, identity, atol=1e-12, rtol=1e-12)
    canonical = frames.transpose(-1, -2) @ matrix @ frames
    magnitude = torch.linalg.vector_norm(pos, dim=-1)
    expected = .001 * identity.clone()
    expected[:, 0, 0] += (magnitude / (magnitude + 1e-8)).square()
    torch.testing.assert_close(canonical, expected, atol=1e-12, rtol=1e-12)
    exp_canonical = frames.transpose(-1, -2) @ torch.matrix_exp(matrix) @ frames
    torch.testing.assert_close(exp_canonical, torch.matrix_exp(expected), atol=2e-12, rtol=2e-12)
    # Verify the mathematical orthogonal-congruence/log equivalence independently.
    rng = np.random.default_rng(727)
    raw = rng.normal(size=(3, 3))
    tangent = (raw + raw.T) / 2
    orthogonal, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    spd = scipy.linalg.expm(tangent)
    lhs = scipy.linalg.logm(orthogonal @ spd @ orthogonal.T)
    rhs = orthogonal @ tangent @ orthogonal.T
    np.testing.assert_allclose(lhs, rhs, atol=3e-12, rtol=3e-12)
    html = (HERE / 'spd_sheaf_icml2026_v1.html').read_text()
    assert 'Sheaf Neural Networks on SPD Manifolds' in html
    sources = [Path(__file__), HERE / 'SPD_SHEAF_AUTHOR_SOURCE_V1.json',
        HERE / 'spd_sheaf_icml2026_v1.html'] + [Path(row['file']) for row in records]
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        author_commit='2c21b53efe9ce1f078c305bd9a4f3ecda57a9833',
        unchanged_selected_author_coordinate_methods_executed=list(names),
        synthetic_nodes=13, dropout_disabled_scope=True, nondegenerate_frames_only=True,
        frame_orthogonality_error=float((frames.transpose(-1, -2) @ frames - identity).abs().max()),
        expected_diagonal_error=float((canonical - expected).abs().max()),
        diagonal_variation_from_author_norm_epsilon=float((expected - expected[0]).abs().max()),
        exponential_canonicalization_error=float((exp_canonical - torch.matrix_exp(expected)).abs().max()),
        independent_orthogonal_log_congruence_error=float(np.max(np.abs(lhs - rhs))),
        icml2026_poster_title_verified=True,
        original_sheaf_code_eigen_clipping_and_rescaling_are_additional_nonlinearities=True,
        full_network_or_training_reproduced=False, measured_performance_claim=False,
        no_coordinate_or_physical_covariance_input_for_this_project=True,
        author_license_file_present=False, formal_neural_fits=0, evaluation_labels_read=False,
        source_sha256={str(path.resolve()): base.digest(path) for path in sources}))
    base.write_json(destination, result)
    print('SPD_SHEAF_COORDINATE_AUDIT', result['expected_diagonal_error'],
          result['diagonal_variation_from_author_norm_epsilon'], result['independent_orthogonal_log_congruence_error'])


if __name__ == '__main__':
    main()
