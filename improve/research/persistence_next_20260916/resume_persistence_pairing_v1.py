"""Resume confirmation after a missing JSON-reader namespace, without editing fits.

The frozen executor calls common.base.read_json, but common.base is the
non_tabular_experiment wrapper, where that helper is absent. A process-local
alias to the existing tabular_foundation reader repairs only that I/O lookup.
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

from improve import anchor_spectral_persistence_experiment_v1 as experiment
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base
from improve.research.persistence_next_20260916 import run_persistence_pairing_queue_v1 as frozen_queue

HERE = Path(__file__).resolve().parent
OUT = frozen_queue.OUT
PLAN = HERE / 'PERSISTENCE_PAIRING_RECOVERY_PROTOCOL_V1.json'


def install_reader():
    module = experiment.common.base
    assert module.__name__ == 'improve.non_tabular_experiment'
    assert not hasattr(module, 'read_json')
    bindings = {name: id(value) for name, value in vars(module).items()}
    module.read_json = base.read_json
    assert all(id(vars(module)[name]) == value for name, value in bindings.items())
    assert set(vars(module)) - set(bindings) == {'read_json'}
    assert module.read_json is base.read_json
    return module


def prepare():
    if PLAN.exists():
        raise ValueError('Preserve registered recovery protocol')
    queue = base.check_seal(base.read_json(HERE / 'PERSISTENCE_PAIRING_QUEUE_PROTOCOL_V1.json'))
    assert queue['source_sha256'] == base.digest(frozen_queue.__file__)
    for path, digest in queue['input_sha256'].items():
        assert base.digest(path) == digest, path
    p = experiment.common.protocol(OUT)
    assessment = base.check_seal(base.read_json(OUT / 'SCREEN_ASSESSMENT.json'))
    assert assessment['declared_gate']
    selection = base.read_json(OUT / 'SELECTION.json')
    assert selection['protocol_sha256'] == base.digest(OUT / 'protocol.json')
    records = {}
    for path in sorted((OUT / 'runs').glob('*/*/*/*/result.json')):
        record = experiment.common.verify(path.parent, base.digest(OUT / 'protocol.json'))
        assert record['seed'] == 666
        records[str(path)] = base.digest(path)
    assert len(records) == 8
    assert not list((OUT / 'runs').glob('*/*/*/s0')) and not list((OUT / 'runs').glob('*/*/*/s1'))
    errors = [OUT / 'confirm_MiDrug.log', OUT / 'confirm_LncDrug.log', HERE / 'persistence_pairing_queue_v1.log']
    for path in errors[:2]:
        content = path.read_text()
        assert "AttributeError: module 'improve.non_tabular_experiment' has no attribute 'read_json'" in content
        assert 'TRAIN_SPECTRAL_PERSISTENCE' not in content
    module = install_reader()
    try:
        assert module.read_json(OUT / 'SELECTION.json') == selection
    finally:
        delattr(module, 'read_json')
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='persistence_pairing',
        original_queue_protocol_sha256=queue['sha256'], source_sha256=base.digest(__file__),
        frozen_training_executor_sha256=base.digest(experiment.__file__),
        missing_api='improve.non_tabular_experiment.read_json', replacement_api='improve.tabular_foundation.read_json',
        existing_runtime_bindings_unchanged=True, json_reader_matches_existing_selection=True,
        repair_applies_only_before_any_confirmation_fit=True, previous_confirmation_neural_fits=0,
        selected_learning_rates={kind: value['lr'] for kind, value in selection['chosen'].items()},
        confirmation_seeds=p['confirm_seeds'], expected_new_confirmation_neural_fits=4,
        expected_new_confirmation_identity_records=4, no_training_or_prediction_function_changed=True,
        original_files_unchanged=True, input_sha256={
            **queue['input_sha256'], **records,
            **{str(path): base.digest(path) for path in errors + [
                OUT / 'SELECTION.json', OUT / 'SCREEN_ASSESSMENT.json',
                Path(experiment.__file__), Path(frozen_queue.__file__), Path(base.__file__)]}},
        confirmed_final_innovations=0))
    base.write_json(PLAN, result)
    print('PERSISTENCE_PAIRING_RECOVERY_FROZEN', result['sha256'], flush=True)


def verify_plan():
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Declared deterministic CuBLAS environment required')
    plan = base.check_seal(base.read_json(PLAN))
    assert plan['source_sha256'] == base.digest(__file__)
    for path, digest in plan['input_sha256'].items():
        assert base.digest(path) == digest, path
    experiment.common.protocol(OUT)
    return plan


def worker(task):
    verify_plan()
    install_reader()
    experiment.run(OUT, task, 'cuda:1', 'confirm')


def resume():
    plan = verify_plan()
    destination = HERE / 'PERSISTENCE_PAIRING_QUEUE_RESULT_V1.json'
    if destination.exists():
        raise ValueError('Preserve completed queue result')
    processes, streams = [], []
    for task in ('MiDrug', 'LncDrug'):
        stream = (OUT / ('confirm_' + task + '_recovery_v1.log')).open('a', buffering=1)
        streams.append(stream)
        command = [sys.executable, '-u', '-m', 'improve.research.persistence_next_20260916.resume_persistence_pairing_v1',
                   'worker', '--task', task]
        processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
    codes = [process.wait() for process in processes]
    for stream in streams:
        stream.close()
    if any(codes):
        raise RuntimeError('Recovered confirmation workers failed: ' + str(codes))
    classification = frozen_queue.classify('confirm')
    filename = 'HISTORICAL_ASSESSMENT.json'
    with (OUT / 'confirm_assessment_recovery_v1.log').open('a', buffering=1) as stream, redirect_stdout(stream):
        if (OUT / filename).exists():
            result = base.check_seal(base.read_json(OUT / filename))
        else:
            result = audit.assess(OUT, 'confirm')
    if not result['declared_gate']:
        stop = OUT / 'HISTORICAL_STOP_DECISION.json'
        if not stop.exists():
            base.write_json(stop, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                decision='Stop this frozen version before partition or outer fitting', stage='confirm',
                primary_kind=result['primary_kind'], assessment_sha256=base.digest(OUT / filename),
                performance_gate=result['performance_gate'], mechanism_gate=result['primary_mechanism_gate'],
                offset_gate=result['offset_null_gate'], confirmed_innovations=0, default_replaced=False)))
    counts = base.check_seal(base.read_json(classification))
    result_record = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        queue_protocol_sha256=plan['original_queue_protocol_sha256'], family='persistence_pairing', stage='confirm',
        historical_gate=result['declared_gate'], assessment_sha256=base.digest(OUT / filename),
        classification_sha256=base.digest(classification), new_neural_fits=counts['cumulative_new_neural_fits'],
        new_identity_records=counts['cumulative_new_identity_records'], adopted_neural_fits=2,
        adopted_identity_records=2, partition_verification_pending=bool(result['declared_gate']),
        prior_parent_outcomes_seen=True, independent_screen=False, confirmed_final_innovations=0,
        existing_default_replaced=False, recovery_protocol_sha256=plan['sha256'],
        recovery_source_sha256=base.digest(__file__), original_training_source_unchanged=True))
    base.write_json(destination, result_record)
    base.write_json(HERE / 'PERSISTENCE_PAIRING_RECOVERY_RESULT_V1.json', base.seal(dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(), recovery_protocol_sha256=plan['sha256'],
        queue_result_sha256=base.digest(destination), confirmation_neural_fits=4, confirmation_identity_records=4,
        original_training_prediction_selection_and_gates_unchanged=True,
        added_runtime_binding_only='non_tabular_experiment.read_json = tabular_foundation.read_json',
        input_sha256_after={path: base.digest(path) for path in plan['input_sha256']},
        declared_gate=result['declared_gate'], confirmed_final_innovations=0)))
    print('PERSISTENCE_PAIRING_RECOVERY_COMPLETE', result['performance_gate'], result['primary_mechanism_gate'],
          result['offset_null_gate'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'resume', 'worker'))
    parser.add_argument('--task', choices=('MiDrug', 'LncDrug'))
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'resume':
        resume()
    else:
        if args.task is None:
            parser.error('worker requires --task')
        worker(args.task)
