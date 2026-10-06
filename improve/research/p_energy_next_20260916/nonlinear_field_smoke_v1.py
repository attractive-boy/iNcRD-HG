"""Training-only real-size nonlinear field solve, stationarity and saved replay."""
from datetime import datetime, timezone
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from improve import nonlinear_pair_field_v1 as core
from improve import anchor_priority_graph_v1 as reference
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
ROOT = reference.ROOT
OUT = ROOT / 'results/nonlinear_pair_field_smoke_v1_20260916'


def main():
    dest = HERE / 'NONLINEAR_FIELD_SMOKE_V1.json'
    if dest.exists() or OUT.exists():
        raise ValueError('Preserve previous nonlinear-field short-check evidence')
    unit_path = HERE / 'NONLINEAR_FIELD_UNIT_V1.json'
    unit = base.check_seal(base.read_json(unit_path))
    assert unit['success'] and unit['tests'] == 5
    for path, digest in unit['source_sha256'].items():
        assert base.digest(path) == digest, path
    source = ROOT / 'results/anchor_priority_morphological_alignment_v1_20260916/inputs/MiDrug/arrays.npz'
    a = reference.nested.load_selected(source, core.TRAIN_KEYS)
    expected, manifest = reference.offset_arrays(ROOT / 'results/nested_anchor_v2_20260915/historical',
                                                'MiDrug', 'historical', a)
    np.testing.assert_array_equal(a['anchor_oof'], expected)
    OUT.mkdir(parents=True)
    p = core.parameters()
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats('cuda:1')
    start = time.monotonic()
    print('NONLINEAR_FIELD_TRAINING_ONLY_START', datetime.now(timezone.utc).isoformat(), flush=True)
    fields, traces, graph_audit, operator = core.fit(a, 'cuda:1', p)
    seconds = time.monotonic() - start
    arrays = {kind: value.cpu().numpy() for kind, value in fields.items()}
    np.savez_compressed(OUT / 'fields.npz', **arrays)
    base.write_json(OUT / 'traces.json', traces)
    base.write_json(OUT / 'graph.json', graph_audit)
    with np.load(OUT / 'fields.npz') as saved:
        reloaded = {kind: torch.as_tensor(saved[kind], device='cuda:1') for kind in core.KINDS}
    train = torch.as_tensor(a['pairs_train'][0] * int(a['nd']) + a['pairs_train'][1], device='cuda:1')
    y = torch.as_tensor(a['y_train'], device='cuda:1', dtype=torch.float64)
    offset = torch.as_tensor(a['anchor_oof'], device='cuda:1', dtype=torch.float64)
    query = a['pairs_train'][:, :157]
    checks = []
    for kind, exponent in (('quadratic', 2.), ('full', 3.)):
        value, gradient = core.value_and_gradient(reloaded[kind], operator, train, y, offset,
                                                exponent, p['regularization'], p['ridge'])
        maximum = float(gradient.abs().max())
        assert maximum <= p['gradient_tolerance']
        assert abs(float(value) - traces[kind][-1]['objective']) < 1e-8
        normal = core.predict(fields[kind], query, int(a['nd']), p['residual_clip']).cpu().numpy()
        reverse = core.predict(reloaded[kind], query[:, ::-1].copy(), int(a['nd']), p['residual_clip']).cpu().numpy()[::-1]
        chunks = np.concatenate([core.predict(reloaded[kind], query[:, i:i+37], int(a['nd']),
            p['residual_clip']).cpu().numpy() for i in range(0, query.shape[1], 37)])
        np.testing.assert_array_equal(normal, reverse)
        np.testing.assert_array_equal(normal, chunks)
        field = reloaded[kind]
        known = torch.zeros(operator.nodes, dtype=torch.bool, device='cuda:1')
        known[train] = True
        # A positive graph and zero-centered ridge obey this discrete maximum
        # principle away from the labeled nodes, up to the stationarity bound.
        slack = p['gradient_tolerance'] / p['ridge']
        lower = min(float(field[train].min()), 0.) - slack
        upper = max(float(field[train].max()), 0.) + slack
        assert float(field[~known].min()) >= lower and float(field[~known].max()) <= upper
        checks.append(dict(kind=kind, exponent=exponent, maximum_gradient=maximum,
            newton_steps=len(traces[kind]) - 1,
            cg_steps=sum(row.get('cg', {}).get('steps', 0) for row in traces[kind]),
            objective=float(value), replay_max_abs=0., discrete_maximum_principle=True,
            field_minimum=float(field.min()), field_maximum=float(field.max())))
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='nonlinear_pair_field',
        parameters=p, device='cuda:1', seconds=seconds,
        peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1') / 2**30,
        graph=graph_audit, checks=checks, new_training_only_numerical_solves=2,
        new_neural_fits=0, formal_fits=0, performance_metrics_computed=False, evaluation_labels_read=False,
        source_sha256={str(path): base.digest(path) for path in (Path(__file__), Path(core.__file__),
            ROOT / 'test_nonlinear_pair_field_v1.py')},
        input_sha256={str(path): base.digest(path) for path in (source, manifest, unit_path)},
        files={name: base.digest(OUT / name) for name in ('fields.npz', 'traces.json', 'graph.json')}))
    base.write_json(dest, result)
    print('NONLINEAR_FIELD_SMOKE_COMPLETE', seconds, result['peak_allocated_gib'], checks, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if OUT.exists() and not (OUT / 'FAILURE.json').exists():
            base.write_json(OUT / 'FAILURE.json', base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                traceback=traceback.format_exc(), source_sha256=base.digest(__file__),
                performance_metrics_computed=False, evaluation_labels_read=False, formal_fits=0)))
        raise
