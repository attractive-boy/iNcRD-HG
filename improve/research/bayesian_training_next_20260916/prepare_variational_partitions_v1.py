"""Promote the confirmed feature-gradient hypothesis to fixed development splits."""
from datetime import datetime,timezone
from pathlib import Path

from improve import anchor_variational_validate_v1 as validation
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = validation.ROOT/'results/mechanism_validation_20260915'


def main():
    target = HERE/'VARIATIONAL_FEATURE_LEARNING_PARTITION_QUEUE_PROTOCOL_V1.json'
    if target.exists():
        raise ValueError('Preserve partition protocol')
    unit_path = HERE/'VARIATIONAL_PARTITION_UNIT_V1.json'
    unit = base.check_seal(base.read_json(unit_path))
    if not unit['passed'] or unit['tests_run'] != 3:
        raise ValueError('Partition implementation checks failed')
    for path,digest in unit['source_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Unit-checked source changed')
    history_path = HERE/'VARIATIONAL_FEATURE_QUEUE_RESULT_V1.json'
    history = base.check_seal(base.read_json(history_path))
    if history['stage'] != 'confirm' or not history['historical_gate']:
        raise ValueError('History failed confirmation')
    family = 'variational_feature_learning'
    validation.freeze(OUT,family)
    dp,cp = validation.checked(OUT,family)
    inputs = [Path(__file__),unit_path,history_path,HERE/'VARIATIONAL_FEATURE_SMOKE_V1.json',
              HERE/'VARIATIONAL_PARTITION_EXECUTOR_DERIVATION_V1.json',OUT/'candidates'/(cp['label']+'.json')]
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),family=family,
        source_sha256=base.digest(HERE/'run_variational_partitions_v1.py'),
        input_sha256={str(path):base.digest(path) for path in inputs},device='cuda:1',
        candidate_sha256=cp['sha256'],recipes=cp['recipes'],split_seeds=dp['split_seeds'],
        expected_new_neural_fits=9,expected_identity_records=9,partition_adoption=False,
        training_inputs_only=True,prediction_uses_aligned_evaluation_anchor=True,
        all_query_quadrature_reference_order=128,quadrature_tolerance=3e-5,
        this_candidate_partition_outcomes_seen=False,previous_candidates_on_same_partitions=True,
        outer_requires_all_partition_gates=True,confirmed_final_innovations=0,existing_default_replaced=False))
    base.write_json(target,queue)
    print('VARIATIONAL_PARTITIONS_FROZEN',queue['sha256'],cp['recipes'],flush=True)


if __name__ == '__main__':
    main()
