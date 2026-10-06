"""Stop the inherited diffusion follow-up at the first failed declared gate."""
from datetime import datetime, timezone
from pathlib import Path

from improve import graph_signal_priority_v1 as study


def main():
    base, out = study.base, study.OUT
    target = study.HERE / 'GRAPH_SIGNAL_PRIORITY_QUEUE_RESULT_V1.json'
    if target.exists():
        raise ValueError('Preserve the completed queue')
    p = study.protocol(out)
    if base.digest(Path(__file__)) != p['input_sha256'][str(Path(__file__).resolve())]:
        raise ValueError('Queue source changed')
    assessments = {}
    for run_stage, report_stage, filename in (
            ('screen', 'screen', 'SCREEN_ASSESSMENT.json'),
            ('confirm', 'historical', 'HISTORICAL_ASSESSMENT.json')):
        study.run(out, run_stage, 'cuda:1')
        if (out / filename).exists():
            assessment = base.check_seal(base.read_json(out / filename))
        else:
            assessment = study.assess(out, report_stage)
        if assessment['protocol_sha256'] != p['sha256']:
            raise ValueError('Assessment identity changed')
        assessments[report_stage] = dict(path=str(out / filename),
            sha256=base.digest(out / filename), declared_gate=assessment['declared_gate'])
        if not assessment['declared_gate']:
            break
    report = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=p['sha256'], source_sha256=base.digest(__file__),
        assessments=assessments, stopped_stage=report_stage,
        next_stage_eligible=bool(report_stage == 'historical' and assessment['declared_gate']),
        new_joint_neural_fits=assessment['new_joint_neural_fits'],
        adopted_joint_neural_fits=assessment['adopted_joint_neural_fits'],
        identity_scoring_records=assessment['identity_scoring_records'],
        new_outer_fits=0, confirmed_final_innovations=0, existing_default_replaced=False))
    base.write_json(target, report)
    print('PRIORITY_DIFFUSION_QUEUE_COMPLETE', report, flush=True)


if __name__ == '__main__':
    main()
