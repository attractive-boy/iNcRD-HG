"""Recompute saved diffusion compositions and gates without modifying source reports."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from improve import graph_signal_priority_v1 as study


def main():
    out, base = study.OUT, study.base
    target = study.HERE / 'GRAPH_SIGNAL_PRIORITY_CONSOLIDATION_V1.json'
    if target.exists():
        raise ValueError('Preserve existing consolidation')
    p = study.protocol(out)
    old_reports = {str(out / name): base.digest(out / name) for name in
                   ('SCREEN_ASSESSMENT.json', 'HISTORICAL_ASSESSMENT.json')}
    primary, controls, count = [], {k: [] for k in study.CONTROLS}, {'new': 0, 'adopted': 0}
    rows, identities = [], []
    for repetition in p['repetitions']:
        seed = repetition['reference_seed']
        for variant in study.VARIANTS:
            directory = out / 'runs/MiDrug' / ('s' + str(seed)) / variant
            record = base.read_json(directory / 'result.json')
            study.verify_files(directory, record)
            calibration = base.read_json(directory / 'calibration.json')
            with np.load(directory / 'scores.npz', allow_pickle=False) as z:
                for readout in study.READOUTS:
                    value = study.old.apply_calibration(z[readout + '_fold_val'], calibration[readout]).mean(0)
                    np.testing.assert_array_equal(value, z[readout + '_calibrated'])
                    composed = z['anchor'] + p['composition_weight'] * value
                    np.testing.assert_array_equal(composed, z[readout + '_composed'])
                    new = study.metrics.metrics(z['y'], composed, z['pairs'])
                    reference = study.metrics.metrics(z['y'], z['anchor'], z['pairs'])
                    row = dict(seed=seed, kind=variant + '_' + readout, metrics=new,
                               delta={m: new[m] - reference[m] for m in ('auc', 'ap')})
                    rows.append(row)
                    if row['kind'] == study.PRIMARY:
                        primary.append(row)
                    elif row['kind'] in controls:
                        controls[row['kind']].append(row)
                for fold in range(5):
                    fd = directory / ('fold_' + str(fold))
                    fr = base.read_json(fd / 'result.json')
                    study.verify_files(fd, fr)
                    if fr['protocol_sha256'] != p['sha256'] or fr['config'] != study.configuration(variant):
                        raise ValueError('Recipe mismatch')
                    trace = base.read_json(fd / 'trace.json')
                    if trace[-1]['epoch'] != 200 or any(not np.isfinite(x['training_reconstruction_loss']) for x in trace):
                        raise ValueError('Wrong or nonfinite training trace')
                    with np.load(fd / 'scores.npz', allow_pickle=False) as fz:
                        ix = np.flatnonzero(z['folds'] == fold)
                        np.testing.assert_array_equal(ix, fz['oof_indices'])
                        for readout in study.READOUTS:
                            np.testing.assert_array_equal(z[readout + '_oof'][ix], fz[readout + '_oof'])
                            np.testing.assert_array_equal(z[readout + '_fold_val'][fold], fz[readout + '_val'])
                    if fr['adopted']:
                        if fr['replay_max_abs'] != 0 or fr['training_steps'] != 0:
                            raise ValueError('Incorrect adopted record')
                        for name in ('result', 'scores', 'model'):
                            source = Path(fr['adoption']['source_result']).parent / (name + ('.json' if name == 'result' else '.npz' if name == 'scores' else '.pt'))
                            if base.digest(source) != fr['adoption']['source_' + name + '_sha256']:
                                raise ValueError('Adoption source changed')
                        count['adopted'] += 1
                    else:
                        if fr['training_steps'] != 200 or not fr['trained_neural_model']:
                            raise ValueError('Incorrect new fit record')
                        count['new'] += 1
        directory = out / 'runs/LncDrug' / ('s' + str(seed))
        record = base.read_json(directory / 'result.json')
        study.verify_files(directory, record)
        with np.load(directory / 'scores.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['anchor'], z['composed'])
        identities.append(dict(seed=seed, training_steps=record['training_steps'], calibration_fits=record['calibration_fits']))
    deltas = {m: float(np.mean([r['delta'][m] for r in primary])) for m in ('auc', 'ap')}
    means = {m: float(np.mean([r['metrics'][m] for r in primary])) for m in ('auc', 'ap')}
    signs = {m: sum(r['delta'][m] > 0 for r in primary) for m in ('auc', 'ap')}
    contrasts = {kind: {m: means[m] - float(np.mean([r['metrics'][m] for r in cell]))
                        for m in ('auc', 'ap')} for kind, cell in controls.items()}
    performance = all(deltas[m] > 0 and signs[m] >= 2 for m in ('auc', 'ap'))
    mechanism = all(v > 0 for c in contrasts.values() for v in c.values())
    assessment = base.check_seal(base.read_json(out / 'HISTORICAL_ASSESSMENT.json'))
    for key, value in (('mean_metrics', means), ('mean_delta', deltas), ('positive_count', signs), ('controls', contrasts)):
        if assessment['MiDrug'][key] != value:
            raise ValueError('Stored historical assessment mismatch: ' + key)
    if (assessment['performance_gate'], assessment['mechanism_gate'], assessment['declared_gate']) != (performance, mechanism, performance and mechanism):
        raise ValueError('Gate mismatch')
    if count != {'new': 30, 'adopted': 15} or any(r['training_steps'] or r['calibration_fits'] for r in identities):
        raise ValueError('Count or identity mismatch')
    if any(base.digest(path) != digest for path, digest in old_reports.items()):
        raise ValueError('Source report was modified')
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=base.digest(__file__), protocol_sha256=p['sha256'],
        original_report_sha256=old_reports, source_reports_unchanged=True,
        mean_metrics=means, mean_delta=deltas, positive_count=signs, contrasts=contrasts,
        performance_gate=performance, mechanism_gate=mechanism,
        new_joint_neural_fits=count['new'], adopted_joint_neural_fits=count['adopted'],
        identity_records=identities, all_oof_rows_checked=True,
        all_saved_calibration_compositions_replayed=True, old_adoption_sources_unchanged=True,
        new_fits_in_consolidation=0, new_inference_in_consolidation=0,
        next_stage_eligible=False, confirmed_final_innovations=0))
    base.write_json(target, result)
    print('GRAPH_SIGNAL_PRIORITY_CONSOLIDATED', result['sha256'], count, flush=True)


if __name__ == '__main__':
    main()
