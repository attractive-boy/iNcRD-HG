"""Append the failed copula partition study to a new immutable consolidation."""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from improve import anchor_copula_validate_v1 as validation
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = validation.ROOT / 'results/mechanism_validation_20260915'


def main():
    record_path = HERE / 'COPULA_PARTITION_CONSOLIDATION_V1.json'
    dest = OUT / 'CONSOLIDATED_RESULTS_V3_20260916.csv'
    if record_path.exists() or dest.exists():
        raise ValueError('Use a new consolidation version')
    _, candidate = validation.checked(OUT, 'bernoulli_copula')
    report = OUT / 'reports' / candidate['label']
    assessment = base.read_json(report / 'ASSESSMENT.json')
    for name, digest in assessment['artifacts'].items():
        if base.digest(report / name) != digest:
            raise ValueError('Partition report artifact changed')
    for name, digest in assessment['result_sha256'].items():
        if base.digest(name) != digest:
            raise ValueError('Partition fit record changed')
    metrics = pd.read_csv(report / 'metrics.csv')
    differences = pd.read_csv(report / 'contrasts.csv')
    rows = []
    for task in ('MiDrug', 'LncDrug'):
        values = metrics[(metrics.task == task) & (metrics.kind == 'full') &
                         (metrics.score == 'anchor_composed')]
        delta = differences[(differences.task == task) & (differences.control == 'anchor')]
        if len(values) != 3 or len(delta) != 3:
            raise ValueError('Incomplete fixed partition report')
        rows.append(dict(label=candidate['label'], name='强参考残差＋二元边 Copula', task=task,
            **{m: float(values[m].mean()) for m in ('auc', 'ap', 'macro_drug_ap')},
            **{'delta_' + m: float(delta[m].mean()) for m in ('auc', 'ap')},
            **{m + '_wins': int((delta[m] > 0).sum()) for m in ('auc', 'ap')},
            **{'min_delta_' + m: float(delta[m].min()) for m in ('auc', 'ap')},
            fresh_gate=assessment['additional_development_gate'],
            performance_gate=assessment['performance_gate'], mechanism_gate=assessment['primary_mechanism_gate'],
            offset_null_gate=assessment['offset_null_gate']))
    parent = OUT / 'CONSOLIDATED_RESULTS_V2_20260916.csv'
    old = pd.read_csv(parent)
    if candidate['label'] in set(old.label):
        raise ValueError('Candidate is already consolidated')
    combined = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    combined.to_csv(dest, index=False)
    stop_path = Path(candidate['parent']) / 'FIXED_PARTITION_STOP_DECISION_V1.json'
    if not assessment['additional_development_gate']:
        if stop_path.exists():
            raise ValueError('Preserve previous stop decision')
        base.write_json(stop_path, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
            decision='Stop this frozen copula version; do not fit outer models',
            assessment_sha256=base.digest(report / 'ASSESSMENT.json'),
            primary_kind=candidate['primary_kind'], performance_gate=False, mechanism_gate=False,
            new_outer_fits=0, confirmed_final_innovations=0, existing_default_replaced=False)))
    grouped = differences.groupby(['task', 'control'])[['auc', 'ap']].mean()
    record = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=base.digest(__file__), prior_csv_sha256=base.digest(parent),
        new_csv=str(dest), new_csv_sha256=base.digest(dest), rows=rows,
        contrasts=[dict(task=task, control=control, **values.to_dict())
                   for (task, control), values in grouped.iterrows()],
        assessment_sha256=base.digest(report / 'ASSESSMENT.json'),
        uncertainty_sha256=base.digest(report / 'endpoint_uncertainty.json'),
        new_historical_neural_fits=16, new_partition_neural_fits=12, new_outer_neural_fits=0,
        preceding_partition_neural_fit_count=264, total_partition_neural_fit_count=276,
        exact_reference_adoptions_and_smoke_fits_excluded=True,
        confirmed_final_innovations=0, existing_default_replaced=False))
    base.write_json(record_path, record)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print('CONSOLIDATED_COPULA', len(combined), flush=True)


if __name__ == '__main__':
    main()
