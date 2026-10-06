"""Freeze SOAP's partition recipe only after read-only confirmation checks."""
from datetime import datetime, timezone
from pathlib import Path

from improve import anchor_eigenbasis_validate_v2 as validation
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = validation.ROOT / 'results/mechanism_validation_20260915'


def main():
    queue_path = HERE / 'EIGENBASIS_ADAM_PARTITION_QUEUE_PROTOCOL_V2.json'
    if queue_path.exists():
        raise ValueError('Preserve frozen partition queue')
    unit_path = HERE / 'EIGENBASIS_PARTITION_UNIT_V2.json'
    unit = base.check_seal(base.read_json(unit_path))
    if not unit['passed'] or unit['tests_run'] != 4:
        raise ValueError('Partition implementation checks failed')
    for path, digest in unit['source_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Unit-tested source changed')
    history_path = HERE / 'EIGENBASIS_QUEUE_RESULT_V1.json'
    history = base.check_seal(base.read_json(history_path))
    if history['stage'] != 'confirm' or not history['historical_gate']:
        raise ValueError('Historical confirmation failed')
    validation.freeze(OUT, 'eigenbasis_adam')
    dp, cp = validation.checked(OUT, 'eigenbasis_adam')
    sources = [unit_path, history_path, HERE/'EIGENBASIS_SMOKE_V1.json',
               HERE/'PARTITION_EXECUTOR_DERIVATION_V2.json', Path(__file__),
               OUT/'candidates/anchor_priority_eigenbasis_adam_v1.json']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        family='eigenbasis_adam', source_sha256=base.digest(HERE/'run_eigenbasis_partitions_v2.py'),
        input_sha256={str(path):base.digest(path) for path in sources}, device='cuda:1',
        candidate_sha256=cp['sha256'], split_seeds=dp['split_seeds'], recipes=cp['recipes'],
        parameter_updates_are_fixed=True, initialization_gradients_per_neural_fit=1,
        new_neural_fits_expected=9, identity_scoring_records_expected=9,
        partition_fit_adoption=False, previous_candidate_partition_outcomes_seen=True,
        this_candidate_partition_outcomes_seen=False, outer_requires_all_partition_gates=True,
        confirmed_final_innovations=0, existing_default_replaced=False))
    base.write_json(queue_path, queue)
    print('FROZEN_EIGENBASIS_PARTITIONS', queue['sha256'], cp['recipes'], flush=True)


if __name__ == '__main__':
    main()
