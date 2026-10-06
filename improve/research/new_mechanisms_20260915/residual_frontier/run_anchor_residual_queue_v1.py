"""Run four already declared historical studies, with two GPU-1 workers.

Do not change recipes, select new primaries, or advance a failed gate.
This coordinator does not create neural/data recipes; those must be frozen.
"""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from improve import anchor_residual_graph_v1 as experiment
from improve import assess_declared_graph as audit
from improve import anchor_residual_validate_v1 as validation
from improve import tabular_foundation as base


def run_stage(out, family, stage):
    processes, streams = [], []
    for task in ('MiDrug', 'LncDrug'):
        stream = (out / (stage + '_' + task + '.log')).open('a', buffering=1)
        streams.append(stream)
        command = [sys.executable, '-m', 'improve.anchor_residual_graph_v1', stage,
                   '--out', str(out), '--family', family, '--task', task, '--device', 'cuda:1']
        processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in processes]
    for stream in streams:
        stream.close()
    if any(codes):
        raise RuntimeError('Study workers failed: ' + str((family, stage, codes)))


def main():
    families = ('geometry', 'state_space', 'block', 'bicentric')
    for family in families:
        out = experiment.ROOT / 'results' / ('anchor_residual_' + family + '_v1_20260915')
        experiment.common.protocol(out)
    for family in families:
        out = experiment.ROOT / 'results' / ('anchor_residual_' + family + '_v1_20260915')
        print('START_SCREEN', family, datetime.now(timezone.utc).isoformat(), flush=True)
        run_stage(out, family, 'screen')
        with (out / 'screen_assessment.log').open('a', buffering=1) as stream, redirect_stdout(stream):
            if not (out / 'SELECTION.json').exists():
                experiment.select(out)
            if (out / 'SCREEN_ASSESSMENT.json').exists():
                experiment.common.all_rows(out)
                result = base.check_seal(base.read_json(out / 'SCREEN_ASSESSMENT.json'))
            else:
                result = audit.assess(out, 'screen')
        print('SCREEN_GATES', family, result['performance_gate'], result['primary_mechanism_gate'],
              result['offset_null_gate'], flush=True)
        if result['declared_gate']:
            run_stage(out, family, 'confirm')
            with (out / 'confirm_assessment.log').open('a', buffering=1) as stream, redirect_stdout(stream):
                if (out / 'HISTORICAL_ASSESSMENT.json').exists():
                    experiment.common.all_rows(out)
                    result = base.check_seal(base.read_json(out / 'HISTORICAL_ASSESSMENT.json'))
                else:
                    result = audit.assess(out, 'confirm')
            print('CONFIRM_GATES', family, result['performance_gate'], result['primary_mechanism_gate'],
                  result['offset_null_gate'], flush=True)
            if result['declared_gate']:
                label = 'anchor_residual_' + family + '_v1'
                candidate = validation.shared.data_source.OUT / 'candidates' / (label + '.json')
                if not candidate.exists():
                    validation.freeze(validation.shared.data_source.OUT, family)
                print('HISTORICAL_PASSED_PARTITIONS_PENDING', family, flush=True)
        if not result['declared_gate']:
            name = 'HISTORICAL_STOP_DECISION.json' if result['stage'] == 'confirm' else 'SCREEN_STOP_DECISION.json'
            assessment = 'HISTORICAL_ASSESSMENT.json' if result['stage'] == 'confirm' else 'SCREEN_ASSESSMENT.json'
            dest = out / name
            if not dest.exists():
                base.write_json(dest, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                    decision='Stop this frozen version before partition fitting',
                    primary_kind=result['primary_kind'], stage=result['stage'],
                    assessment_sha256=base.digest(out / assessment),
                    performance_gate=result['performance_gate'], mechanism_gate=result['primary_mechanism_gate'],
                    offset_gate=result['offset_null_gate'], confirmed_innovations=0, default_replaced=False)))
        print('FINISHED_HISTORICAL', family, datetime.now(timezone.utc).isoformat(), flush=True)


if __name__ == '__main__':
    main()
