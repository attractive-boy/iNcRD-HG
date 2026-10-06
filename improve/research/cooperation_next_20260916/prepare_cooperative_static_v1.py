"""Freeze a narrower cooperative hypothesis and audit stage-one artifact reuse."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

from improve import anchor_priority_graph_v1 as experiment
from improve import cooperative_graph_v3 as core
from improve import adopt_cooperative_static_v1 as adoption
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    root = experiment.ROOT
    old_core = root / 'results/cooperative_local_core_v1_20260916'
    old = root / 'results/anchor_priority_cooperative_local_v1_20260916'
    parent = experiment.common.protocol(old_core)
    previous = experiment.common.protocol(old)
    selection = base.read_json(old / 'SELECTION.json')
    smoke_path = HERE / 'COOPERATIVE_STATIC_SMOKE_V1.json'
    smoke = base.check_seal(base.read_json(smoke_path))
    if smoke['performance_metrics_computed'] or smoke['formal_fits'] or len(smoke['fold_audits']) != 40:
        raise ValueError('Invalid smoke boundary')
    if smoke['core_parameters'] != core.parameters():
        raise ValueError('Core parameters changed')
    for path, digest in smoke['source_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Smoke source changed')
    core_out = root / 'results/cooperative_static_core_v1_20260916'
    out = root / 'results/anchor_priority_cooperative_static_v1_20260916'
    queue_path = HERE / 'COOPERATIVE_STATIC_QUEUE_PROTOCOL_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Preserve existing studies')
    source_records = {}
    for task in previous['tasks']:
        for stage in ('screen', 'confirm'):
            for lr, seed in adoption.cells(previous, selection, stage):
                path = old / 'runs' / task / 'static_policy' / ('lr' + str(lr)) / ('s' + str(seed)) / 'result.json'
                experiment.common.verify(path.parent, base.digest(old / 'protocol.json'))
                source_records[str(path)] = base.digest(path)
    additions = ('cooperative_graph_v3.py', 'test_cooperative_graph_v3.py',
        'adopt_cooperative_static_v1.py', 'test_adopt_cooperative_static_v1.py',
        'adopt_cooperative_local_v1.py')
    sources = {**parent['source_sha256'], **{name: base.digest(root / name) for name in additions}}
    input_files = [old_core / 'protocol.json', old / 'protocol.json', old / 'SELECTION.json',
        old / 'HISTORICAL_ASSESSMENT.json', smoke_path, Path(__file__),
        HERE / 'COOPERATIVE_STATIC_ADAPTATION_V1.md', HERE / 'COOPERATIVE_ADAPTATION_V1.md',
        root / 'results/mechanism_validation_20260915/reports/anchor_priority_cooperative_local_v1/ASSESSMENT.json']
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='cooperative_static',
        mechanism_family='cooperative', implementation_module='cooperative_graph_v3',
        tasks=['MiDrug', 'LncDrug'], kinds=list(core.KINDS), primary_kind='full',
        primary_controls=['soft_actions', 'shared_policy'], core_parameters=core.parameters(),
        lnc_policy='identity_strong_reference', identity_records_are_not_neural_fits=True,
        selection='Full is the already observed static_policy. New controls isolate discrete actions and node-specific rates. No evolving-state benefit is claimed. Learning-rate selection is unchanged.',
        source_sha256=sources, inputs={**parent['inputs'], **source_records,
            **{str(path): base.digest(path) for path in input_files}},
        literature=parent['literature'],
        mechanism='Initial-state, node-specific discrete send/receive actions; one cooperative family.',
        same_innovation_family_as_prior_version=True, revision_of=str(old),
        full_historical_and_partition_predictions_previously_observed=True,
        new_control_scores_observed=False, no_independent_screening_claim=True,
        baseline_adoption=dict(source=str(old), source_kind='static_policy', target_kind='full',
            cpu_forward_gradient_rng_equivalence_test_passed=True, result_sha256=source_records,
            expected_screen_neural_adoptions=2, expected_confirm_neural_adoptions=2),
        evaluation='Every original performance, mechanism, fixed-partition and outer gate remains mandatory.',
        smoke=False, original_test_source_access=False, confirmed_innovations=0, existing_default_replaced=False)
    core_out.mkdir(parents=True)
    (core_out / 'source_snapshot').mkdir()
    for name in sources:
        shutil.copy2(root / name, core_out / 'source_snapshot' / name)
    base.write_json(core_out / 'protocol.json', p)
    experiment.prepare(out, 'cooperative_static', root / 'results/nested_anchor_v2_20260915/historical', core_out)
    adoption.adopt(out, 'screen')
    inputs = [core_out / 'protocol.json', out / 'protocol.json', out / 'COOP_STATIC_ADOPTION_SCREEN_V1.json',
              smoke_path, HERE / 'COOPERATIVE_STATIC_ADAPTATION_V1.md']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='cooperative_static',
        source_sha256=base.digest(HERE / 'run_cooperative_static_queue_v1.py'),
        input_sha256={str(path): base.digest(path) for path in inputs}, device='cuda:1',
        stages=['screen', 'confirm'], stop_on_any_failed_gate=True,
        expected_new_neural_fits={'screen': 4, 'confirm_if_eligible': 4},
        expected_adopted_neural_fits={'screen': 2, 'confirm_if_eligible': 2},
        identity_records={'screen': 6, 'confirm_if_eligible': 6},
        adopted_identity_records={'screen': 2, 'confirm_if_eligible': 2},
        same_mechanism_family_as_previous=True, confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('STATIC_COOPERATIVE_FROZEN', base.digest(out / 'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    main()
