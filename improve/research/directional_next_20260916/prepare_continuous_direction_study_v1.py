"""Freeze the declared independent-direction mechanism before performance use."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

from improve import anchor_priority_graph_v1 as experiment
from improve import continuous_direction_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    smoke_path = HERE / 'CONTINUOUS_DIRECTION_SMOKE_V1.json'
    smoke = base.check_seal(base.read_json(smoke_path))
    unit_path = HERE / 'CONTINUOUS_DIRECTION_UNIT_V1.json'
    unit = base.check_seal(base.read_json(unit_path))
    assert smoke['formal_fits'] == 0 and not smoke['performance_metrics_computed']
    assert not smoke['evaluation_labels_read']
    assert smoke['core_parameters'] == core.parameters() and len(smoke['fold_audits']) == 40
    assert unit['passed'] and unit['tests_run'] == 5
    for source in (smoke['source_sha256'], unit['input_sha256']):
        for path, digest in source.items():
            if base.digest(path) != digest:
                raise ValueError('Pre-performance evidence changed: ' + path)
    core_out = experiment.ROOT / 'results/continuous_direction_core_v1_20260916'
    out = experiment.ROOT / 'results/anchor_priority_continuous_direction_v1_20260916'
    queue_path = HERE / 'CONTINUOUS_DIRECTION_QUEUE_PROTOCOL_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Preserve previous studies; use a new version')
    sources = ['continuous_direction_graph_v1.py', 'test_continuous_direction_graph_v1.py',
        'anchor_direction_experiment_v1.py', 'sharpness_graph_v1.py',
        'test_anchor_priority_graph_v1.py', 'overlap_block_models.py',
        'overlap_block_experiment.py', 'non_tabular_experiment.py',
        'tabular_foundation.py', 'graph_prediction_precision.py',
        'nested_anchor_inputs_v2.py', 'residual_graph_experiment.py',
        'anchor_priority_graph_v1.py', 'assess_declared_graph.py']
    inputs = [smoke_path, unit_path, HERE / 'COED_ADAPTATION_V1.md',
        HERE / 'continuous_direction_smoke_v1.py', HERE / 'continuous_direction_unit_v1.log', Path(__file__),
        HERE / 'run_continuous_direction_queue_v1.py', HERE / 'coed_tree_v1.json',
        HERE / 'DIRECTIONAL_AUTHOR_TREES_V1.json', HERE / 'DIRECTIONAL_AUTHOR_SOURCE_V1.json',
        HERE / 'DIRECTIONAL_FULLTEXT_FETCH_V1.json', HERE / 'COED_PDF_VERIFICATION_V1.json',
        HERE / 'coed_paper_v1.pdf', HERE / 'coed_paper_v1.txt', HERE / 'openalex_learning_v3.json']
    for entry in base.read_json(HERE / 'DIRECTIONAL_AUTHOR_SOURCE_V1.json')['records']:
        if entry['key'] != 'coed':
            continue
        path = Path(entry['file']).resolve()
        if entry.get('status') != 'downloaded' or base.digest(path) != entry['sha256']:
            raise ValueError('Author source integrity failed')
        inputs.append(path)
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='continuous_direction',
        implementation_module='continuous_direction_graph_v1', tasks=['MiDrug', 'LncDrug'],
        kinds=list(core.KINDS), primary_kind='full', primary_controls=['symmetric_weights', 'fixed_direction'],
        core_parameters=core.parameters(), lnc_policy='identity_strong_reference',
        identity_records_are_not_neural_fits=True,
        selection='Primary and both controls fixed before new performance. Two neural learning rates, fixed theta lr=0.01, no theta decay, projected angles, 400 updates and every 20-step checkpoint. No architecture sweep.',
        source_sha256={name: base.digest(experiment.ROOT / name) for name in sources},
        inputs={str(path.resolve()): base.digest(path) for path in inputs},
        literature=dict(title='Improving Graph Neural Networks by Learning Continuous Edge Directions',
            arxiv='2410.14109v1', version_date='2024-10-18', venue='Preprint',
            author_commit='5e88be92c22754f022add14ba3b8c5a5e22603c5', full_text_obtained=True,
            evidence='Read original method and pinned released code; squared-mass code differs from the inspected paper; no publication claim beyond preprint'),
        mechanism='Independent projected edge direction parameters with separated in/out messages; symmetric learned affinities and fixed neutral directions are mandatory controls',
        required_executor='improve.anchor_direction_experiment_v1',
        direction_update_gate='Every step projects angles to [0,pi/2]; theta lr=0.01 and decay=0; unchanged fixed control',
        evaluation='Historical seed 666 screen, seeds 0/1 confirmation, three fixed development partitions, then frozen outer recheck; stop this version at any failed gate.',
        no_new_performance_observed=True, smoke=False, original_test_source_access=False,
        prior_failures_preserved=True, existing_default_replaced=False, confirmed_innovations=0)
    core_out.mkdir(parents=True)
    (core_out / 'source_snapshot').mkdir()
    for name in sources:
        shutil.copy2(experiment.ROOT / name, core_out / 'source_snapshot' / name)
    base.write_json(core_out / 'protocol.json', p)
    experiment.prepare(out, 'continuous_direction', experiment.ROOT / 'results/nested_anchor_v2_20260915/historical', core_out)
    frozen_inputs = [core_out / 'protocol.json', out / 'protocol.json', smoke_path, unit_path,
                     HERE / 'COED_ADAPTATION_V1.md']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='continuous_direction',
        source_sha256=base.digest(HERE / 'run_continuous_direction_queue_v1.py'),
        input_sha256={str(path): base.digest(path) for path in frozen_inputs},
        device='cuda:1', stages=['screen', 'confirm'], stop_on_any_failed_gate=True,
        expected_new_neural_fits={'screen': 6, 'confirm_if_eligible': 6},
        expected_identity_records={'screen': 6, 'confirm_if_eligible': 6},
        adopted_neural_fits=0, identity_records_count_as_neural_fits=False,
        confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('CONTINUOUS_DIRECTION_FROZEN', base.digest(out / 'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    main()
