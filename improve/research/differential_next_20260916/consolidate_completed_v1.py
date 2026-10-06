"""Read-only verification of completed studies; write new consolidation files."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from improve import anchor_plain_validate_v3 as validation

base, ROOT = validation.base, validation.ROOT
HERE = Path(__file__).resolve().parent


def new_json(path, value):
    if path.exists():
        raise ValueError('Preserve existing consolidation: ' + str(path))
    base.write_json(path, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        consolidation_source_sha256=base.digest(__file__), **value)))


def historical(family, research, prefix):
    parent = ROOT / ('results/anchor_priority_' + family + '_v1_20260916')
    result = validation.read_historical_assessment(parent)
    ph = base.digest(parent / 'protocol.json')
    records = [validation.experiment.common.verify(path.parent, ph)
               for path in sorted((parent / 'runs').glob('*/*/*/*/result.json'))]
    counts = validation.historical_counts(records)
    queue = base.check_seal(base.read_json(research / (prefix + '_QUEUE_RESULT_V1.json')))
    assert queue['assessment_sha256'] == base.digest(parent / 'HISTORICAL_ASSESSMENT.json')
    assert queue['new_neural_fits'] == counts['new_neural_fits']
    assert queue['adopted_neural_fits'] == counts['adopted_neural_fits']
    assert queue['historical_gate'] == result['declared_gate']
    value = dict(family=family, stage='confirm', protocol_sha256=ph,
        assessment_sha256=base.digest(parent / 'HISTORICAL_ASSESSMENT.json'),
        queue_result_sha256=base.digest(research / (prefix + '_QUEUE_RESULT_V1.json')),
        all_stored_metrics_recomputed=True, source_assessment_unchanged=True,
        metrics=result['means'], primary_minus_controls=result['primary_minus_controls'],
        accounting=counts, performance_gate=result['performance_gate'],
        mechanism_gate=result['primary_mechanism_gate'], offset_gate=result['offset_null_gate'],
        historical_gate=result['declared_gate'], confirmed_final_innovations=0,
        existing_default_replaced=False)
    if family == 'cooperative_static':
        assert not result['declared_gate']
        value.update(mechanism_family='cooperative',
            next_stage='stop_before_partition_freeze',
            full_screen_and_partitions_previously_observed=True,
            new_partition_fits=0, new_outer_fits=0,
            unsealed_partition_drafts_unused=True)
    else:
        value['historical_promotion_now_resolved_by_partition_result'] = True
    new_json(research / (prefix + '_HISTORICAL_CONSOLIDATION_V1.json'), value)
    print('HISTORICAL_VERIFIED', family, counts, result['declared_gate'], flush=True)
    return counts


def partitions(counts):
    out = ROOT / 'results/mechanism_validation_20260915'
    family = 'differential_attention'
    dp, cp = validation.checked(out, family)
    report = out / 'reports' / cp['label']
    assessment = base.read_json(report / 'ASSESSMENT.json')
    assert assessment['candidate_sha256'] == cp['sha256']
    assert assessment['data_protocol_sha256'] == dp['sha256']
    for name, digest in assessment['artifacts'].items():
        assert base.digest(report / name) == digest
    for name, digest in assessment['result_sha256'].items():
        path = Path(name)
        assert base.digest(path) == digest
        r = base.read_json(path)
        validation.shared.verify_artifacts(path.parent, r)
        z = validation.experiment.common.load(path.parent / 'scores.npz')
        for score, expected in r['metrics'].items():
            actual = validation.experiment.common.base.metrics(z['y'], z[score], z['pairs'])
            for metric in ('auc', 'ap', 'macro_drug_ap'):
                assert abs(actual[metric] - expected[metric]) < 1e-12
        if r['task'] == 'LncDrug':
            assert r['trained_neural_model'] is False
            np.testing.assert_array_equal(z['standalone'], np.zeros_like(z['standalone']))
            np.testing.assert_array_equal(z['anchor_composed'], z['anchor'])
    frame = pd.read_csv(report / 'metrics.csv')
    contrasts = pd.read_csv(report / 'contrasts.csv')
    rows = []
    for task in dp['tasks']:
        scores = frame[(frame.task == task) & (frame.kind == cp['primary_kind']) &
                       (frame.score == 'anchor_composed')]
        delta = contrasts[(contrasts.task == task) & (contrasts.control == 'anchor')]
        assert len(scores) == len(delta) == 3
        rows.append(dict(label=cp['label'], name='差分注意力（作者 V2 运算适配）', task=task,
            auc=float(scores.auc.mean()), ap=float(scores.ap.mean()),
            delta_auc=float(delta.auc.mean()), delta_ap=float(delta.ap.mean()),
            auc_wins=int((delta.auc > 0).sum()), ap_wins=int((delta.ap > 0).sum()),
            min_delta_auc=float(delta.auc.min()), min_delta_ap=float(delta.ap.min()),
            macro_drug_ap=float(scores.macro_drug_ap.mean()),
            fresh_gate=assessment['additional_development_gate'],
            performance_gate=assessment['performance_gate'],
            mechanism_gate=assessment['primary_mechanism_gate'],
            offset_null_gate=assessment['offset_null_gate']))
    previous = out / 'CONSOLIDATED_RESULTS_V5_20260916.csv'
    target = out / 'CONSOLIDATED_RESULTS_V6_20260916.csv'
    if target.exists():
        raise ValueError('Preserve existing table')
    old = pd.read_csv(previous)
    assert len(old) == 30 and cp['label'] not in set(old.label)
    result = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    assert len(result) == 32
    result.to_csv(target, index=False)
    old_count = base.check_seal(base.read_json(ROOT / 'research/feature_interaction_next_20260916/'
        'COOPERATIVE_LOCAL_PARTITION_CONSOLIDATION_V1.json'))
    assert old_count['table_sha256'] == base.digest(previous)
    assert assessment['new_neural_fits'] == assessment['identity_scoring_records'] == 6
    uncertainty = base.read_json(report / 'endpoint_uncertainty.json')
    summary = pd.read_csv(report / 'summary.csv').to_dict('records')
    new_json(HERE / 'DIFFERENTIAL_ATTENTION_PARTITION_CONSOLIDATION_V1.json',
        dict(family=family, candidate_sha256=cp['sha256'],
            assessment_sha256=base.digest(report / 'ASSESSMENT.json'),
            table_sha256=base.digest(target), previous_table_sha256=base.digest(previous),
            rows=rows, all_stored_metrics_recomputed=True, source_assessment_unchanged=True,
            mean_contrasts=summary, uncertainty=uncertainty,
            new_partition_neural_fits=6, new_partition_identity_records=6,
            cumulative_fixed_partition_neural_fits=old_count['cumulative_fixed_partition_neural_fits'] + 6,
            historical_accounting=counts, new_reference_fits=0, new_outer_fits=0,
            confirmed_final_innovations=0, existing_default_replaced=False,
            decision='Stop this frozen candidate before outer recheck', all_reports_preserved=True))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__':
    counts = historical('differential_attention', HERE, 'DIFFERENTIAL_ATTENTION')
    historical('cooperative_static', ROOT / 'research/cooperation_next_20260916', 'COOPERATIVE_STATIC')
    partitions(counts)
