"""Append completed fixed-partition results without changing the old CSV."""
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from improve import anchor_residual_validate_v2 as validation


HERE = Path(__file__).resolve().parent
QUEUE = HERE.parent / 'new_mechanisms_20260915/residual_frontier'
NAMES = {'geometry': '强参考残差＋双曲注意力', 'block': '强参考残差＋变分群组',
         'bicentric': '强参考残差＋双中心子图'}
base = validation.base


def main():
    out = validation.shared.data_source.OUT
    dest = out / 'CONSOLIDATED_RESULTS_V2_20260916.csv'
    assessment_path = HERE / 'PARTITION_CONSOLIDATION_V1.json'
    if dest.exists() or assessment_path.exists():
        raise ValueError('Preserve existing consolidation')
    record = base.check_seal(base.read_json(QUEUE / 'PARTITION_QUEUE_RESULT_V2.json'))
    old = out / 'CONSOLIDATED_RESULTS.csv'
    with old.open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    old_rows = len(rows)
    reports, new_rows, fits = {}, [], 0
    for family in NAMES:
        dp, cp = validation.checked(out, family)
        label, primary = cp['label'], cp['primary_kind']
        rd = out / 'reports' / label
        path = rd / 'ASSESSMENT.json'
        assert base.digest(path) == record['results'][family]['assessment_sha256']
        ar = base.read_json(path)
        assert ar['candidate_sha256'] == cp['sha256']
        assert ar['data_protocol_sha256'] == dp['sha256']
        for name, digest in ar['artifacts'].items():
            assert base.digest(rd / name) == digest
        for result_path, digest in ar['result_sha256'].items():
            result_path = Path(result_path)
            assert base.digest(result_path) == digest
            result = base.read_json(result_path)
            validation.shared.verify_artifacts(result_path.parent, result)
            scores = validation.experiment.common.load(result_path.parent / 'scores.npz')
            for name, expected in result['metrics'].items():
                actual = validation.experiment.common.base.metrics(scores['y'], scores[name], scores['pairs'])
                for metric in ('auc', 'ap', 'macro_drug_ap'):
                    assert abs(actual[metric] - expected[metric]) < 1e-12
            fits += 1
        expected_fits = sum(len(cp['recipes'][task]) * len(dp['split_seeds']) for task in dp['tasks'])
        assert len(ar['result_sha256']) == expected_fits
        metrics = pd.read_csv(rd / 'metrics.csv')
        contrasts = pd.read_csv(rd / 'contrasts.csv')
        for task in dp['tasks']:
            m = metrics[(metrics.task == task) & (metrics.kind == primary) & (metrics.score == 'anchor_composed')]
            delta = contrasts[(contrasts.task == task) & (contrasts.control == 'anchor')]
            assert len(m) == len(delta) == 3
            row = dict(label=label, name=NAMES[family], task=task,
                auc=m.auc.mean(), ap=m.ap.mean(), delta_auc=delta.auc.mean(), delta_ap=delta.ap.mean(),
                auc_wins=int((delta.auc > 0).sum()), ap_wins=int((delta.ap > 0).sum()),
                min_delta_auc=delta.auc.min(), min_delta_ap=delta.ap.min(),
                macro_drug_ap=m.macro_drug_ap.mean(), fresh_gate=ar['additional_development_gate'],
                performance_gate=ar['performance_gate'], mechanism_gate=ar['primary_mechanism_gate'],
                offset_null_gate=ar['offset_null_gate'])
            rows.append(row)
            new_rows.append(row)
        reports[family] = dict(assessment=str(path), sha256=base.digest(path),
            performance_gate=ar['performance_gate'], mechanism_gate=ar['primary_mechanism_gate'],
            overall_gate=ar['additional_development_gate'], formal_partition_fits=expected_fits,
            uncertainty=base.read_json(rd / 'endpoint_uncertainty.json'))
    assert old_rows == 18 and fits == 48
    outer = base.check_seal(base.read_json(QUEUE / 'OUTER_QUEUE_RESULT_V1.json'))
    assert not any(x['overall_gate'] for x in reports.values())
    assert outer['eligible_families'] == [] and outer['results'] == {}
    with dest.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=base.digest(__file__), old_csv_sha256=base.digest(old),
        new_csv_sha256=base.digest(dest), old_rows=old_rows, new_rows=new_rows,
        reports=reports, new_partition_fits=fits, cumulative_partition_fits=216 + fits,
        new_historical_neural_fits=76, new_outer_neural_fits=0,
        historical_reference_contexts_excluded_from_fit_counts=True,
        outer_skipped_because_all_partition_gates_failed=True,
        full_scores_recomputed=True, old_csv_preserved=True,
        confirmed_final_innovations=0, existing_default_replaced=False,
        limitations=['All partitions are overlapping, previously exposed development data.',
            'Engineering LncDrug tolerance is 0.001, not a statistical noninferiority claim.',
            'Bootstrap intervals condition on selected predictors and exclude selection uncertainty.']))
    base.write_json(assessment_path, result)
    for row in new_rows:
        print(row['label'], row['task'],
              'AUC/AP', f"{row['auc']:.12f}", f"{row['ap']:.12f}",
              'delta', f"{row['delta_auc']:+.12f}", f"{row['delta_ap']:+.12f}",
              'positive splits', row['auc_wins'], row['ap_wins'], flush=True)
    print('PARTITION_CONSOLIDATION_COMPLETE', fits, 'new fits;', len(rows), 'rows', flush=True)


if __name__ == '__main__':
    main()
