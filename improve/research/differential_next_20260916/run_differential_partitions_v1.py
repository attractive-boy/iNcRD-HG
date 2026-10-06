"""Execute an already frozen priority-family fixed-partition validation."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from improve import anchor_plain_validate_v3 as validation
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = validation.ROOT / 'results/mechanism_validation_20260915'


def main(family):
    label = validation.label_for(family)
    protocol_path = HERE / (family.upper() + '_PARTITION_QUEUE_PROTOCOL_V1.json')
    protocol = base.check_seal(base.read_json(protocol_path))
    if protocol['family'] != family or protocol['source_sha256'] != base.digest(__file__):
        raise ValueError('Partition queue changed')
    for path, digest in protocol['input_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Partition queue input changed')
    result_path = HERE / (family.upper() + '_PARTITION_QUEUE_RESULT_V1.json')
    if result_path.exists():
        raise ValueError('Preserve completed partition queue')
    dp, candidate = validation.checked(OUT, family)
    streams, processes = [], []
    for task in ('MiDrug', 'LncDrug'):
        stream = (HERE / (family + '_partitions_' + task + '_v1.log')).open('a', buffering=1)
        streams.append(stream)
        command = [sys.executable, '-u', '-m', 'improve.anchor_plain_validate_v3', 'run',
            '--family', family, '--out', str(OUT), '--task', task, '--device', 'cuda:1']
        processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
    codes = [process.wait() for process in processes]
    for stream in streams:
        stream.close()
    if any(codes):
        raise RuntimeError('Priority partition workers failed: ' + str(codes))
    for task in ('MiDrug', 'LncDrug'):
        for split in dp['split_seeds']:
            for kind, recipe in candidate['recipes'][task].items():
                path = OUT / 'runs' / label / task / str(split) / kind / 'result.json'
                r = base.read_json(path)
                if task == 'MiDrug':
                    if not r['trained_neural_model'] or r['training_steps'] != recipe['epochs'] or r['gradient_evaluations'] != recipe['epochs']:
                        raise ValueError('Neural fixed-epoch budget mismatch')
                elif r['trained_neural_model'] or any(r[k] != 0 for k in
                        ('epoch', 'parameters', 'trainable_parameters', 'training_steps', 'gradient_evaluations')):
                    raise ValueError('Identity scored as a neural fit')
    with (HERE / (family + '_partition_report_v1.log')).open('a', buffering=1) as stream, redirect_stdout(stream):
        validation.report(OUT, family)
    report_path = OUT / 'reports' / label / 'ASSESSMENT.json'
    assessment = base.read_json(report_path)
    expected = len(dp['split_seeds']) * len(candidate['recipes']['MiDrug'])
    if assessment['new_neural_fits'] != expected or assessment['identity_scoring_records'] != expected:
        raise ValueError('Unexpected partition fit/identity count')
    base.write_json(result_path, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        queue_protocol_sha256=protocol['sha256'], candidate_sha256=candidate['sha256'],
        family=family, assessment_sha256=base.digest(report_path),
        performance_gate=assessment['performance_gate'], mechanism_gate=assessment['primary_mechanism_gate'],
        partition_gate=assessment['additional_development_gate'],
        outer_verification_pending=assessment['additional_development_gate'],
        new_partition_fit_count=assessment['new_neural_fits'],
        identity_scoring_records=assessment['identity_scoring_records'],
        confirmed_final_innovations=0, existing_default_replaced=False)))
    print('PRIORITY_FIXED_PARTITIONS_COMPLETE', family, assessment['additional_development_gate'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=tuple(validation.REGISTRY), required=True)
    main(parser.parse_args().family)
