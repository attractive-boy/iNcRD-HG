"""Read-only source audit and one fixed-weight exploratory score diagnostic."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from improve import non_tabular_experiment as metrics
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'results/graph_signal_calibrated_20260914'
REFERENCE = ROOT / 'results/observation_offset_inputs_20260915/MiDrug/arrays.npz'
OUT = HERE / 'SAVED_DIFFUSION_RECHECK_V1.json'
WEIGHT = 0.1


def main():
    if OUT.exists():
        raise ValueError('Preserve the existing diagnostic')
    manifest = {str(REFERENCE): base.digest(REFERENCE)}
    with np.load(REFERENCE, allow_pickle=False) as z:
        a = {k: z[k].copy() for k in ('pairs_train', 'y_train', 'pairs_val', 'y_val',
                                     'anchor_666', 'anchor_0', 'anchor_1')}
    seeds = (666, 0, 1)
    reference = {str(s): metrics.metrics(a['y_val'], a['anchor_' + str(s)], a['pairs_val'])
                 for s in seeds}
    records, models, rows = {}, set(), []
    for variant in ('support', 'annotation', 'shuffled'):
        directory = OLD / 'runs/MiDrug/inner' / variant
        path = directory / 'result.json'
        record = base.read_json(path)
        manifest[str(path)] = base.digest(path)
        records[str(path)] = manifest[str(path)]
        for name, expected in record['files'].items():
            file = directory / name
            if base.digest(file) != expected:
                raise ValueError('Changed source artifact: ' + str(file))
            manifest[str(file)] = expected
            if name.endswith('model.pt'):
                models.add(str(file))
        with np.load(directory / 'scores.npz', allow_pickle=False) as z:
            for old_key, new_key in (('pairs', 'pairs_val'), ('y', 'y_val'),
                                     ('pairs_train', 'pairs_train'), ('y_train', 'y_train')):
                np.testing.assert_array_equal(z[old_key], a[new_key])
            for key, expected in record['metrics'].items():
                current = metrics.metrics(z['y'], z[key], z['pairs'])
                for name, value in expected.items():
                    if abs(current[name] - value) > 1e-12:
                        raise ValueError('Stored metric mismatch')
            for readout in ('reverse', 'one_step'):
                score = z[readout + '_calibrated']
                standalone = metrics.metrics(z['y'], score, z['pairs'])
                old_rows = [metrics.metrics(z['y'], z[readout + '_gnn_' + str(s)], z['pairs'])
                            for s in seeds]
                new_rows = [metrics.metrics(z['y'], a['anchor_' + str(s)] + WEIGHT * score,
                                            z['pairs']) for s in seeds]
                row = dict(variant=variant, readout=readout, standalone=standalone,
                    old_half_gnn_mean={m: float(np.mean([r[m] for r in old_rows]))
                                       for m in ('auc', 'ap')},
                    exploratory_anchor_composed={str(s): new_rows[i] for i, s in enumerate(seeds)},
                    exploratory_mean={m: float(np.mean([r[m] for r in new_rows]))
                                      for m in ('auc', 'ap')},
                    exploratory_mean_delta={m: float(np.mean([new_rows[i][m] - reference[str(s)][m]
                                                              for i, s in enumerate(seeds)]))
                                            for m in ('auc', 'ap')})
                rows.append(row)
    primary = next(r for r in rows if r['variant'] == 'annotation' and r['readout'] == 'reverse')
    controls = {}
    for variant, readout in (('support', 'reverse'), ('shuffled', 'reverse'), ('annotation', 'one_step')):
        row = next(r for r in rows if r['variant'] == variant and r['readout'] == readout)
        controls[variant + '_' + readout] = {
            prefix: {m: primary[prefix][m] - row[prefix][m] for m in ('auc', 'ap')}
            for prefix in ('old_half_gnn_mean', 'exploratory_mean')}
    if any(base.digest(path) != digest for path, digest in records.items()):
        raise ValueError('Source record was changed')
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=base.digest(__file__), input_sha256=manifest, records=rows,
        reference=reference, controls=controls, fixed_exploratory_weight=WEIGHT,
        coefficient_search_performed=False, new_fits=0, new_inference=0,
        old_neural_models_verified=len(models), old_metrics_recomputed=True,
        old_result_records_unchanged=True, task='MiDrug', original_test_access=False,
        scope='Existing historical development predictions only; exploratory follow-up prompted by an old failed candidate. Three reference initializations share one five-fold diffusion ensemble, so they are not independent diffusion repeats.',
        lnc_scores_changed=False, old_shared_two_task_failure_unchanged=True,
        confirmed_innovations=0, eligible_as_formal_confirmation=False))
    base.write_json(OUT, result)
    print({k: primary[k] for k in ('variant', 'readout', 'standalone', 'old_half_gnn_mean',
                                   'exploratory_mean', 'exploratory_mean_delta')}, flush=True)
    print(controls, flush=True)
    print('SEALED', OUT, result['sha256'], flush=True)


if __name__ == '__main__':
    main()
