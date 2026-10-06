"""Run the frozen loopy path study, stopping at the first failed declared gate."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from improve import anchor_residual_graph_v2 as experiment
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base


HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/anchor_residual_loopy_path_v2_20260916'


def stage(name):
    processes, streams = [], []
    for task in ('MiDrug', 'LncDrug'):
        stream = (OUT / (name + '_' + task + '.log')).open('a', buffering=1)
        streams.append(stream)
        command = [sys.executable, '-u', '-m', 'improve.anchor_residual_graph_v2', name,
            '--family', 'loopy_path', '--out', str(OUT), '--task', task, '--device', 'cuda:1']
        processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
    codes = [process.wait() for process in processes]
    for stream in streams:
        stream.close()
    if any(codes):
        raise RuntimeError('Loopy workers failed: ' + str((name, codes)))


def main():
    frozen = base.check_seal(base.read_json(HERE / 'LOOPY_QUEUE_PROTOCOL_V1.json'))
    if frozen['source_sha256'] != base.digest(__file__):
        raise ValueError('Frozen loopy coordinator changed')
    for path, digest in frozen['input_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Frozen queue input changed')
    experiment.common.protocol(OUT)
    result_path = HERE / 'LOOPY_QUEUE_RESULT_V1.json'
    if result_path.exists():
        raise ValueError('Preserve the completed queue result')
    for name in ('screen', 'confirm'):
        print('START_LOOPY_STAGE', name, datetime.now(timezone.utc).isoformat(), flush=True)
        stage(name)
        with (OUT / (name + '_assessment.log')).open('a', buffering=1) as stream, redirect_stdout(stream):
            if name == 'screen' and not (OUT / 'SELECTION.json').exists():
                experiment.select(OUT)
            filename = 'SCREEN_ASSESSMENT.json' if name == 'screen' else 'HISTORICAL_ASSESSMENT.json'
            if (OUT / filename).exists():
                experiment.common.all_rows(OUT)
                result = base.check_seal(base.read_json(OUT / filename))
            else:
                result = audit.assess(OUT, name)
        print('LOOPY_GATES', name, result['performance_gate'], result['primary_mechanism_gate'],
              result['offset_null_gate'], flush=True)
        if not result['declared_gate']:
            stop = OUT / ('SCREEN_STOP_DECISION.json' if name == 'screen' else 'HISTORICAL_STOP_DECISION.json')
            if not stop.exists():
                base.write_json(stop, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                    decision='Stop this frozen version before additional partition fitting', stage=name,
                    primary_kind=result['primary_kind'], assessment_sha256=base.digest(OUT / filename),
                    performance_gate=result['performance_gate'], mechanism_gate=result['primary_mechanism_gate'],
                    offset_gate=result['offset_null_gate'], confirmed_innovations=0, default_replaced=False)))
            break
    base.write_json(result_path, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        queue_protocol_sha256=frozen['sha256'], stage=name, historical_gate=result['declared_gate'],
        assessment_sha256=base.digest(OUT / filename),
        partition_verification_pending=bool(result['declared_gate']),
        confirmed_final_innovations=0, existing_default_replaced=False)))
    print('LOOPY_QUEUE_COMPLETE', name, result['declared_gate'], flush=True)


if __name__ == '__main__':
    main()
