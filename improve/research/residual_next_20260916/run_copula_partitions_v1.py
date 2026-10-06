"""Run the frozen copula on existing fixed development partitions."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from improve import anchor_copula_validate_v1 as validation
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = validation.ROOT / 'results/mechanism_validation_20260915'
FAMILY = 'bernoulli_copula'
LABEL = 'anchor_residual_' + FAMILY + '_v1'


def main():
    protocol = base.check_seal(base.read_json(HERE / 'COPULA_PARTITION_QUEUE_PROTOCOL_V1.json'))
    if protocol['source_sha256'] != base.digest(__file__):
        raise ValueError('Partition queue changed')
    for path, digest in protocol['input_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Partition queue input changed')
    result_path = HERE / 'COPULA_PARTITION_QUEUE_RESULT_V1.json'
    if result_path.exists():
        raise ValueError('Preserve completed partition queue')
    numerical = base.check_seal(base.read_json(HERE / 'COPULA_POSTERIOR_AUDIT_V1.json'))
    if not numerical['passed']:
        raise ValueError('Copula numerical posterior audit failed')
    if not (OUT / 'candidates' / (LABEL + '.json')).exists():
        validation.freeze(OUT, FAMILY)
    _, candidate = validation.checked(OUT, FAMILY)
    streams, processes = [], []
    for task in ('MiDrug', 'LncDrug'):
        stream = (HERE / ('copula_partitions_' + task + '_v1.log')).open('a', buffering=1)
        streams.append(stream)
        command = [sys.executable, '-u', '-m', 'improve.anchor_copula_validate_v1', 'run',
            '--family', FAMILY, '--out', str(OUT), '--task', task, '--device', 'cuda:1']
        processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
    codes = [process.wait() for process in processes]
    for stream in streams:
        stream.close()
    if any(codes):
        raise RuntimeError('Fixed copula partition workers failed: ' + str(codes))
    with (HERE / 'copula_partition_report_v1.log').open('a', buffering=1) as stream, redirect_stdout(stream):
        validation.report(OUT, FAMILY)
    report_path = OUT / 'reports' / LABEL / 'ASSESSMENT.json'
    assessment = base.read_json(report_path)
    base.write_json(result_path, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        queue_protocol_sha256=protocol['sha256'], candidate_sha256=candidate['sha256'],
        assessment_sha256=base.digest(report_path), performance_gate=assessment['performance_gate'],
        mechanism_gate=assessment['primary_mechanism_gate'],
        partition_gate=assessment['additional_development_gate'],
        outer_verification_pending=assessment['additional_development_gate'],
        new_partition_fit_count=12, confirmed_final_innovations=0, existing_default_replaced=False)))
    print('COPULA_FIXED_PARTITIONS_COMPLETE', assessment['additional_development_gate'], flush=True)


if __name__ == '__main__':
    main()
