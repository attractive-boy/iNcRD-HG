"""Recompute completed historical assessments without rewriting source reports."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from improve import anchor_priority_graph_v1 as shared
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base


def consolidate(out, target, stage):
    out, target = Path(out).resolve(), Path(target).resolve()
    if target.exists():
        raise ValueError('Preserve completed consolidation')
    p = shared.common.protocol(out)
    ph = base.digest(out / 'protocol.json')
    name = 'SCREEN_ASSESSMENT.json' if stage == 'screen' else 'HISTORICAL_ASSESSMENT.json'
    source = out / name
    before = base.digest(source)
    stored = base.check_seal(base.read_json(source))
    selection = base.read_json(out / 'SELECTION.json')
    if (stored['stage'] != stage or stored['protocol_sha256'] != ph
            or selection['protocol_sha256'] != ph
            or stored['selection_sha256'] != base.digest(out / 'SELECTION.json')
            or stored['assessment_source_sha256'] != base.digest(audit.__file__)):
        raise ValueError('Historical provenance changed')
    all_records, selected_records, selected_files = [], {}, {}
    seeds = [p['screen_seed']] + (p['confirm_seeds'] if stage == 'confirm' else [])
    identity, adopted_identity, neural, adopted_neural = 0, 0, 0, 0
    for path in sorted((out / 'runs').glob('*/*/*/*/result.json')):
        r = shared.common.verify(path.parent, ph)
        z = shared.common.load(path.parent / 'scores.npz')
        for score, expected in {**r['metrics'], **r['offset_metrics']}.items():
            actual = shared.common.base.metrics(z['y'], z[score], z['pairs'])
            for metric in ('auc', 'ap', 'macro_drug_ap'):
                if abs(actual[metric] - expected[metric]) >= 1e-12:
                    raise ValueError('Stored metric changed')
        adopted = r.get('adopted_without_new_training', False)
        if adopted and (r['new_training_steps'] != 0
                or base.digest(r['adoption_source_result']) != r['adoption_source_result_sha256']):
            raise ValueError('Invalid adoption record')
        if r['trained_neural_model']:
            if r['task'] != 'MiDrug' or r['training_steps'] != p['epochs']:
                raise ValueError('Unexpected neural fitting record')
            neural += not adopted
            adopted_neural += adopted
        else:
            if r['task'] != 'LncDrug' or any(r[k] != 0 for k in
                    ('epoch', 'parameters', 'trainable_parameters', 'training_steps', 'gradient_evaluations')):
                raise ValueError('Unexpected identity record')
            np.testing.assert_array_equal(z['standalone'], np.zeros_like(z['standalone']))
            np.testing.assert_array_equal(z['anchor_composed'], z['anchor'])
            identity += 1
            adopted_identity += adopted
        all_records.append(dict(path=str(path), sha256=base.digest(path),
                                trained_neural_model=r['trained_neural_model'], adopted=adopted))
        if r['lr'] == selection['chosen'][r['kind']]['lr'] and r['seed'] in seeds:
            selected_records[r['task'], r['kind'], r['seed']] = r
            selected_files[str(path)] = base.digest(path)
    if selected_files != stored['result_sha256']:
        raise ValueError('Selected result set changed')
    recomputed = audit.evaluate(p, selected_records, seeds)
    for key, expected in recomputed.items():
        if stored.get(key) != expected:
            raise ValueError('Assessment recomputation differs: ' + key)
    if base.digest(source) != before:
        raise ValueError('Source assessment was unexpectedly modified')
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        consolidation_source_sha256=base.digest(__file__), family=p['family'], stage=stage,
        protocol_sha256=ph, assessment_sha256=before,
        all_stored_metrics_recomputed=True, source_assessment_unchanged=True,
        metrics=stored['means'], contrasts=stored['primary_minus_controls'],
        selected_learning_rates={k: v['lr'] for k, v in selection['chosen'].items()},
        new_neural_fits=neural, adopted_neural_fits=adopted_neural,
        identity_scoring_records=identity, adopted_identity_records=adopted_identity,
        performance_gate=stored['performance_gate'], mechanism_gate=stored['primary_mechanism_gate'],
        offset_gate=stored['offset_null_gate'], declared_gate=stored['declared_gate'],
        records=all_records, next_stage_eligible=stored['declared_gate'],
        confirmed_final_innovations=0, existing_default_replaced=False))
    base.write_json(target, result)
    print(p['family'], stage, neural, adopted_neural, identity, stored['declared_gate'], flush=True)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--stage', choices=('screen', 'confirm'), required=True)
    args = parser.parse_args()
    consolidate(args.out, args.target, args.stage)
